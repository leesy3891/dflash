# 배치 스윕 프로파일링 설계안 (`batch` 브랜치)

Context length `S`를 CLI(`--context-length`)로 4k / 8k / 16k / 32k 중 하나로 고정한
상태에서 배치 `B`를 늘려 가며 DFlash와 baseline(`block_size=1`)을 비교한다. 목표는
세 가지다: **throughput**, **baseline 대비 speedup**, 그리고 **메모리 병목 진단**.
메모리 병목은 두 측면으로 나눠 본다. 하나는 무엇이 max B를 막는지(용량, capacity),
다른 하나는 decode step의 시간을 어떤 바이트가 지배하는지(대역폭, bandwidth)다.

측정 원칙은 [PROFILING.md](PROFILING.md), [PROFILING2.md](PROFILING2.md),
[visualization_selective/VISUALIZATION.md](visualization_selective/VISUALIZATION.md)에서
그대로 가져온다. 메모리는 한 시점에서 동시에 분해하고, first draft와 steady draft를
분리하며, sharded 점과는 시간 비교를 하지 않는다. 계산에 쓴 per-token 계수도 모두
`record_selective/`의 B=1 측정에서 가져왔다.

> 요청의 배치 목록 "1, 2, 4, 4, 16"은 **1, 2, 4, 8, 16**으로 해석했다. 32는 §7의 메모리
> 예측상 들어가는 점에서만 추가한다.

## 0. 요약

* **현재 코드는 B=1 전용이다.** 행마다 수락 길이가 다른(ragged) 배치 speculative
  decoding은 batch 차원을 늘리는 것만으로는 되지 않아서, 배치 엔진을 새로 짜야 한다(§4).
* **착수 전에 반드시 풀어야 할 문제가 셋 있다(§2).**
  1. **[확인됨] Qwen3.5-9B의 GDN recurrent state는 rollback되지 않는다.** 따라서 기존
     9B DFlash 기록은 lossless가 아니고, 그 수락 길이와 speedup은 신뢰할 수 없다.
  2. `DynamicCache`의 `torch.cat` 때문에 decode step마다 전체 KV가 복사된다. B·S가
     커지면 이 artifact가 "KV 대역폭 병목"을 부풀린다.
  3. GDN conv record buffer(384 KiB/token)가 prefill 피크에 들어간다. 이것이 9B의
     배치 가능 범위를 가장 크게 좁히는 항이다.
* **설계의 핵심**: 요청마다 따로 prefill과 drafter prefill을 하고, 결과를 슬롯에 넣은 뒤
  배치로 decode한다. KV는 행별 길이를 갖는 static KV와 행별 mask로 관리하고, 9B에는
  GDN replay rollback을 추가한다.
* **통제**: 출력 길이는 EOS를 억제해 **N=256으로 고정**한다(HF `min_new_tokens`와 같은
  의미). 모든 B가 **같은 32개 프롬프트**를 쓴다(paired design).
* **예측**: 8B는 target KV(144 KiB/token)가 한계를 정하며 B·S ≲ 130k tokens까지 된다.
  9B는 §2.3을 고치고 나면 KV가 32 KiB/token에 불과해 32k에서도 B=8까지 들어간다.

## 1. 답하려는 질문

| | 질문 | 답이 나오는 곳 |
| --- | --- | --- |
| Q1 | S를 고정하고 B를 키울 때 throughput과 same-B speedup은 어떻게 변하며, **왜** 그런가? | §5.1, §5.2 (step 비용 분해) |
| Q2 | (용량) 각 (S, B)에서 피크를 무엇이 차지하는가? max B를 정하는 성분은 무엇인가? DFlash는 max B를 얼마나 깎는가? | §5.3, §7 |
| Q3 | (대역폭) decode 각 phase는 HBM-bound, compute-bound, CPU-bound 중 어디에 속하는가? 가중치, KV, KV 복사, GDN state 중 어느 바이트가 시간을 지배하는가? | §5.4 |

## 2. 선결 과제

### 2.1 현재 코드에 박힌 B=1 가정

| 위치 | 가정 |
| --- | --- |
| `model.py:791-794` | `output_ids`가 `(1, …)`이고 `position_ids`를 모든 행이 공유 |
| `model.py:958`, `:1088-1090` | `start` 스칼라 하나로 모든 행을 진행시키고, `_crop_to`로 캐시 전체를 한 길이로 자름 |
| `model.py:1076-1077` | 수락 길이를 `.sum(dim=1)[0].item()`로, bonus를 `[0]`으로 읽음 |
| `model.py:219-249` | `_rejection_sample`이 `[0]`, `.item()`으로 행 하나만 처리 |
| `model.py:1082-1087` | stop 검사가 `output_ids[0, …]` |
| `model.py:1004`, `:1104` | draft의 position slice와 `[:, :produced, :]`가 행 공통 |
| `benchmark.py:651-682` | 프롬프트 하나씩 `encode` → `dflash_generate` |

행마다 수락 길이 `a_b`가 다르므로 cache 길이, position, 다음 context feature 길이가
모두 행마다 달라진다. 이것이 §4.2의 per-row 설계가 필요한 이유다.

### 2.2 [확인됨] Qwen3.5-9B: GDN recurrent state가 rollback되지 않는다

transformers 5.16.1의 `LinearAttentionCacheLayerMixin.crop`(`cache_utils.py:971-996`)은
**`conv_states`만 자르고 `recurrent_states`는 건드리지 않는다**. verify는 16토큰을
처리한 뒤의 state를 `update_recurrent_state`(`:1077-1091`)로 저장한다. 그래서 reject된
토큰의 기여가 state에 남은 채로 다음 step에 전파된다. 24개 GDN 레이어 모두에서,
그리고 매 step마다 이 일이 일어난다.

CPU 재현. GDN 1층과 full-attention 1층짜리 랜덤 모델을 fp32로 만들고,
`dflash.model._make_cache` / `_crop_to`를 그대로 썼다(부록 A):

| 경로 | clean prefix 대비 relative error |
| --- | --- |
| P개 prefill → 3개 추가(reject 없음) | 7.3e-08 (수치 한계) |
| P개 prefill → 8개 verify → 3개만 유지 | **6.7e-03** |

기존 기록에서도 같은 정황이 보인다. 9B에서는 DFlash가 512 토큰 상한까지 가는데
baseline은 수십 토큰에서 멈추는 샘플이 반복된다(4k: 512 vs 20, 512 vs 39 /
8k: 512 vs 31 / 32k: 512 vs 58). 오염된 state가 반복적이고 퇴화한 출력을 만들고,
그런 출력은 draft하기 쉬워 수락이 부풀었을 가능성이 있다. 8B에도 길이 불일치는
있지만, 이는 verify 폭에 따른 bf16 수치 차이 수준으로 설명된다.

**함의.** `record/`와 `record_selective/`에 있는 9B DFlash의 수락 길이(6.9–9.0)와
speedup은 target이 오염된 상태로 측정된 값이다. PROFILING.md에는 "9B가 논문의 27B
drafter보다 높다"는 관찰이 있는데, 이 버그로 설명될 수 있다. 다만 이것은 추정이고,
Phase 0에서 정량화한다. 메모리 수치는 영향을 받지 않는다(state 크기가 같다). 수정은
B=1에서도 필요하다(§4.3).

### 2.3 GDN conv record buffer가 prefill 피크에 들어간다

`activate_past_recording()`을 켠 채로 prefill하면 24개 GDN 레이어가 각자 conv 입력
전체를 보관한다(`cache_utils.py:1072`). 크기는 8192 ch × 2 B = 16 KiB/token/layer이고,
합하면 **384 KiB/token**이다. 64k에서는 24 GiB가 된다. PROFILING2가 "target KV live at
the peak 26.05 GB"라고 본 값의 대부분이 이것이다(24 GiB에 full-attn KV 2 GiB가 더해짐).
prefill 직후의 `crop(0)`은 view만 만들기 때문에 storage를 풀지 못하고, 첫 verify의
`cat`에서야 해제된다. draft sliding cache에서 본 view-retention 현상과 같다.

prefill에는 rollback이 필요 없으니 **recording은 decode에서만 켜면 된다**. §7에서 보듯
이 한 가지 수정으로 9B의 32k max B가 1에서 8로 바뀐다.

### 2.4 `DynamicCache`의 `torch.cat`: step마다 전체 KV를 복사한다

`DynamicLayer.update`(`cache_utils.py:144-145`)는 `torch.cat([keys, new])`로 매 step
레이어의 KV 전체를 새로 할당하고 복사한다. 그래서 step당 KV 트래픽은 attention read
1회에 복사 read+write 2회가 더해진다.

* 추정치(측정 아님): 8B 64k B=1에서 KV가 9 GiB이면 복사가 약 18 GiB/step이고,
  시간으로 약 25 ms다. 측정된 baseline tpot이 68.7 ms이니 작지 않은 비중이다.
* 이 비용은 step당으로 들기 때문에, token당으로 보면 baseline이 DFlash보다 τ배 더
  불리하다. 기존 long-context speedup이 그만큼 유리하게 나왔을 수 있다.
* **결정**: 배치 엔진은 preallocated static KV를 쓴다. B=1에서 `--kv-cache dynamic`
  ablation으로 이 artifact의 크기를 직접 측정하고, 그 결과를 기존 기록과 잇는
  교량으로 남긴다.

## 3. 실험 변수와 통제

| 구분 | 값 |
| --- | --- |
| model | `qwen3-8b`, `qwen3.5-9b` |
| S (CLI로 고정) | 4096, 8192, 16384, 32768. 프로세스 하나가 S 하나를 맡는다 |
| B | 1, 2, 4, 8, 16. 32는 §7에서 fit하는 곳만 |
| mode | `dflash`(block 16), `baseline`(block 1). **같은 엔진, 같은 캐시, 같은 커널**을 쓴다 |

**프롬프트.** S마다 풀 P=32개를 쓴다. 입력 길이가 S와 정확히 같은 것만 남기고, 맞지
않는 것은 버린 뒤 보충한다(`context.py`의 fitter는 ±16까지 허용한다). 모든 B가 같은
32개를 P/B개의 배치로 나눠 처리하므로, B 간 비교는 내용과 수락 분포가 같은 **paired
비교**가 된다. 배치 구성은 seed를 고정해 섞는다.

**출력 길이는 N=256으로 고정한다.** target logits에서 EOS를 `-inf`로 억제한다(HF
`min_new_tokens`와 같은 의미). greedy에서 verify에도 같은 처리가 적용되므로 DFlash는
여전히 lossless다. 억제 전 argmax가 EOS였던 첫 위치(`natural_eos_at`)는 행마다
기록하고, 수락률을 pre-EOS와 post-EOS로 나눠 보고한다.

* 이유: 자연 종료로 두면 출력이 3–512 토큰(p50 6–60)으로 흩어지고, 32k에서는 평균
  12–24 토큰에 그친다(`record_selective/`). 그러면 배치의 실제 occupancy가 대부분
  1–2가 되어 "배치 B에서의 throughput"이 정의되지 않는다. PROFILING.md Caveats에 적힌
  16k/32k 경계 문제도 이 방식으로 함께 풀린다.
* N=256이면 decode 중 S가 늘어나는 폭이 272 토큰 이내라서(4k에서 6.6% 이하) "S
  고정"이라는 가정을 유지할 수 있다.

**공통 설정.** greedy, `--reasoning off`, `--hidden-states selective`, 단일 GPU(sharding
금지), `--prefill-chunk` 없음, `torch.manual_seed(0)`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. 드래프터는 기존 관례대로 두 모드
모두에서 로드한다. 다만 iso-memory 분석에서는 baseline의 피크에서 draft weight를 뺀
값도 함께 계산한다.

**Warmup.** 각 (S, B, mode)마다 실제 shape로 prefill 1회와 decode 8 step을 돌린다.

## 4. 배치 엔진 설계

### 4.1 흐름

```
for each batch of B prompts:
  for each row b (한 행씩, B=1):                                   # prefill 단계
      target prefill (tap으로 선택된 layer hidden만 받음) → first token
      context feature → drafter prefill (first draft forward) → step-0 proposal
      target KV / draft KV / GDN state를 슬롯 b에 직접 기록, context feature 해제
  while any row active:                                            # 배치 decode
      verify (B×16) → 행별 수락 a_b → 행별 전진 L_b += a_b + 1
      [9B] GDN replay rollback
      context feature build (행별 a_b+1개, max 길이로 pad)
      draft forward (B×16) → draft logits → proposals
```

**prefill을 요청 단위로 하는 이유.**

1. S≥4k에서는 prefill이 이미 compute-bound라서, 묶어도 throughput 이득이 작다.
2. 피크가 `B·S·(prefill 항)`에서 `(B−1)·S·(resident 항) + S·(prefill 항)`으로 내려간다
   (§7의 A열 vs B열).
3. padding이 없으므로 GDN left-padding 문제를 피할 수 있다(`apply_mask_to_padding_states`
   는 padding이 recurrence에 주는 영향까지 정확히 지우지 못한다).
4. prefill과 decode를 분리하는 것은 실제 서빙 시스템의 방식과도 같다.

**drafter prefill은 prefill 단계에 넣는다.** PROFILING2 §3이 보인 O(S) first draft
call은 요청당 한 번 하는 작업이므로, 서빙 관점에서는 prefill 비용이다. 이렇게 하면
decode 루프에는 steady draft만 남는다. drafter prefill 시간은 TTFT에 포함하되
`drafter_prefill_s`로 따로 보고한다.

**prefill은 슬롯 b를 가리키는 B=1 view 캐시에 바로 쓴다.** 임시 캐시에 먼저 쓰고
복사하면 행 하나의 KV가 이중으로 잡혀 피크가 S·144 KiB만큼 커진다.

`--prefill-mode batched`는 ablation으로만 둔다(8B 4k, B≤8).

### 4.2 행별 static KV와 행별 mask

**target의 full-attention 레이어.**

* KV를 `(B, H_kv, S+N+16, d)`로 미리 할당하고, 행별 유효 길이 `L_b`를 둔다.
* verify는 `[L_b, L_b+16)`에 쓴다. mask는 행 b의 query i가 key `j < L_b + i + 1`만
  보도록 만든다.
* 수락 후에는 `L_b += a_b+1`만 하면 된다. 남은 슬롯은 다음 step이 덮어쓰므로 crop이
  필요 없다.
* attention은 K/V를 `max_b L_b + 16`까지만 잘라서 호출해 쓰지 않는 꼬리를 읽지 않는다.
  행 간 길이 차이(≤ N+16) 때문에 생기는 masked read 낭비는 `masked_kv_fraction`으로
  기록한다.
* HF 모델과는 이렇게 잇는다. `AttentionInterface.register("dflash_batch", fn)`로
  커스텀 attention을 등록하고, 행별 길이는 모듈 레벨 핸들로 넘긴다. 이는 기존
  `_DRAFT_STAGES`와 같은 방식이다(kwargs로 넘기면 SDPA까지 전달된다). 캐시는 HF Cache
  인터페이스(`update`, `get_seq_length`, linear-attn 메서드들)를 구현한 자체 클래스로
  만든다.

**draft.**

* 행 b에서 새로 들어오는 context 토큰 수를 `n_b = a_b+1`이라 하자. context K/V는
  `[L_b^prev, L_b^prev + n_max)`에 쓰고, noise block은 `[L_b, L_b+16)`에 쓴다.
  `n_max − n_b ≤ 15 < 16`이므로 padding 행이 남긴 쓰레기는 **항상 noise block에
  덮인다**.
* 9B draft의 sliding 레이어 5개는 ring buffer(4096+32 슬롯)로 두고, 슬롯마다 절대
  위치를 저장해 그것으로 mask를 만든다. RoPE는 캐시에 넣기 전에 적용되므로 슬롯
  순서는 상관없다. 결과적으로 draft KV가 S에 비례하지 않는다(steady에서는 full layer
  1개분인 4 KiB/token).

**SDPA backend.** 임의 mask가 있으면 flash backend를 쓸 수 없고 mem-efficient가 쓰인다.
어느 backend가 쓰였는지 kernel 이름으로 기록한다. math backend로 떨어지면 B×H×q×S
크기의 fp32 score가 생겨 메모리와 시간이 모두 달라지기 때문이다.

### 4.3 GDN rollback (Qwen3.5-9B, B=1 포함)

**저장.** verify 직전에 recurrent state 사본을 떠 둔다(행당 24층 × 32×128×128, fp32로
48 MiB). verify 중에는 레이어별 `(q, k, v, g, β)`를 저장한다(행당 약 6–10 MiB).
캡처는 `modeling_qwen3_5.torch_chunk_gated_delta_rule`을 모듈 속성으로 감싸서 한다.

**replay.** 저장한 입력의 앞 `a_b+1` step을 recurrent 규칙으로 다시 적용해 state를
만든다. 행마다 길이가 다른 문제는, `a_b+1` 이후 step에서 `g=0`, `β=0`으로 두면
`S_t = S_{t−1}`인 항등 갱신이 되므로 **배치 호출 한 번**으로 처리된다. causality 덕분에,
수락된 prefix가 각 레이어에 준 입력은 verify에서 계산한 값과 같다. 따라서 결과는
정확하다.

**conv state.** record 모드에서 쌓인 `(kernel−1+16)`개 열 중, 행별 `a_b`에 맞는 창을
gather한다.

**기각한 대안.**

| 대안 | 기각 이유 |
| --- | --- |
| 토큰별 state를 모두 저장 | 행당 384–768 MiB가 들어, B=16이면 6–12 GiB |
| target을 다시 실행 | verify 비용이 2배 |

replay 비용은 `decode: accept/rollback` phase에 넣고, DFlash 고유 overhead로 분류한다.

### 4.4 먼저 끝난 행

N에 도달한 행도 배치에 남겨 계산은 계속하되 결과는 버린다. 이렇게 하면 shape가
고정되고, 정적 배치 서빙의 실제 비용이 그대로 드러난다. step마다 active row 수를
기록한다.

throughput은 두 가지로 잰다.

* **full-occupancy**: 첫 행이 끝나기 전까지의 구간. 배치 B에서의 순수 step 비용을 본다.
* **makespan**: 배치 전체 구간.

baseline은 모든 행이 동시에 끝나므로 두 값이 같다. DFlash에서 둘의 차이가 **straggler
비용**, 즉 행별 수락 길이 분산이 치르는 대가다. continuous batching(끝난 슬롯을 새
요청으로 채우기)은 이번 범위에서 뺀다.

### 4.5 OOM 처리

* 실행 전에 §7의 계수로 피크를 예측하고, 초과하면 `status: "skipped_predicted_oom"`으로
  남긴다.
* static KV는 배치를 시작할 때 할당되므로 KV로 인한 OOM은 곧바로 드러난다. 잡아서
  `status: "oom"`, `oom_phase`, 그때 할당량을 기록하고 다음 점으로 넘어간다.
* 경계 검증용으로 예측 한계 바로 바깥의 점을 하나씩 실행한다(예: 8B 32k B=8).

### 4.6 CLI (안)

```
dflash benchmark transformers --model-preset qwen3-8b --context-length 16384 \
  --batch-sizes 1,2,4,8,16 --fixed-output-tokens 256 --num-prompts 32 \
  --reasoning off --hidden-states selective --record-dir record_batch \
  [--prefill-mode sequential|batched] [--kv-cache static|dynamic] \
  [--profile-kernels N_STEPS] [--no-baseline]
```

한 프로세스가 S 하나에 대해 B 목록을 순회하므로 모델은 한 번만 로드한다. 기록은 점마다
파일 하나(`record_batch/<model>_<S>_b<B>_<stamp>.json`)라서, 중간에 OOM이 나도 앞선
점은 남는다.

### 4.7 파일 변경 범위

| 파일 | 변경 |
| --- | --- |
| `dflash/batch.py` (신규) | BatchCache(static KV, ring buffer, GDN state와 replay), 행별 mask attention, prefill/insert, decode loop, step timer |
| `dflash/model.py` | draft attention이 외부 mask와 캐시 writer를 받도록 수정하고, `_attention_mask`를 행별로 일반화. 기존 `dflash_generate`는 회귀 비교용으로 남긴다 |
| `dflash/benchmark.py`, `dflash/cli.py` | `_run_batch_sweep`과 새 인자 |
| `dflash/record.py` | 배치 요약, step trace, roofline 계산 |
| `queue/run_batch_sweep.sh` | 기존 스크립트처럼 idle GPU만 잡도록 확인 |
| `visualization_batch/` | §9 |
| `tests/` (신규) | tiny 모델로 CPU에서 ragged mask와 GDN replay의 정확성 검증 |

## 5. 관측 지표와 breakdown

### 5.1 Throughput과 speedup의 정의

* decode throughput: `Thr_m(B) = Σ_rows 출력 토큰 / T_decode`. full-occupancy와
  makespan 두 가지를 모두 낸다.
* e2e throughput: `Σ 출력 토큰 / (T_prefill_total + T_decode)`
* per-request TPOT는 `T_decode / N`이다. TTFT는 자기 prefill에 drafter prefill을 더한
  값이고, 배치 안에서 기다린 시간은 `queue_wait_s`로 따로 적는다.

| 이름 | 식 | 의미 |
| --- | --- | --- |
| **same-B speedup** | `Thr_DF(B) / Thr_BL(B)` | 헤드라인. 같은 배치에서 SD가 주는 이득 |
| vs-B1 speedup | `Thr_DF(B) / Thr_BL(1)` | 배치와 SD를 합친 이득 |
| batch scaling efficiency | `Thr_m(B) / (B · Thr_m(1))` | 모드별 배치 확장성 |
| **iso-memory speedup** | `max_{B: peak_DF ≤ M} Thr_DF / max_{B: peak_BL ≤ M} Thr_BL` | 메모리 한도 M에서도 DFlash가 이득인가. baseline은 drafter 몫의 메모리로 B를 더 키울 수 있다 |

iso-memory는 throughput–peak memory Pareto 그림으로 시각화한다(F6).

### 5.2 step 비용 분해: speedup이 왜 변하는가

decode step마다 CUDA event로 다음 항을 잰다.

| 항 | 범위 |
| --- | --- |
| `T_verify` | target forward(DFlash는 B×16, baseline은 B×1) |
| `T_draft` | steady draft forward. 기존의 stage 5개 분해를 유지 |
| `T_ctx` | context feature build |
| `T_accept` | argmax, 비교, 행별 전진, [9B] GDN replay |
| `T_cpu` | `wall − Σ GPU event`. launch overhead와 `.item()` sync |

step당 토큰은 `Σ_b (a_b+1) = B·τ̄_B`이다. 따라서

```
speedup_same-B = τ̄_B · T_step,BL(B) / T_step,DF(B)
               = τ̄_B / (c_verify + c_draft + c_ctx + c_accept + c_cpu),   c_x = T_x / T_step,BL(B)
```

**검증 기준**: 이렇게 기록된 항만으로 측정 speedup을 ±3% 안에서 다시 만들어낼 수
있어야 한다. 그러면 B가 커질 때 어느 `c_x`가 커지는지가 곧 Q1의 답이다. 특히
`c_verify(B,S) = T_verify(B,16) / T_fwd(B,1)`을 본다. 1에 가까우면 verify가 아직
memory-bound이고, 커지면 compute-bound에 들어간 것이다.

### 5.3 용량(capacity) 병목

기존 `PhaseMemory`를 그대로 쓰고 phase만 배치 흐름에 맞게 다시 정의한다.

| 단계 | phase |
| --- | --- |
| prefill (행별) | `prefill: target forward`, `prefill: context-feature build`, `prefill: drafter prefill`, `prefill: insert` |
| decode (배치) | `decode: target verify`, `decode: accept/rollback`, `decode: context-feature build`, `decode: draft forward`, `decode: draft logits` |

component를 몇 개 추가한다. `target_kv_allocated`와 `target_kv_used`(Σ L_b 기준),
`gdn_state`, `gdn_replay_buffer`, `verify_logits`.

**산출물.**

1. 각 (S, B)에서 peak phase와 그 시점의 성분 분해.
2. **per-token 계수 적합.** 각 phase 피크를 `peak = W + (B·S)·α + B·β`로 회귀한다.
   B=1 결과가 PROFILING2 값과 맞는지 확인한다(8B: KV 144, sel 40, act ≈105,
   draft KV 20 KiB/token / 9B: full-attn KV 32, act ≈210 KiB/token).
3. max B. 측정한 OOM 경계와 예측을 비교한다.
4. **max B를 정하는 성분**, 즉 피크에서 B에 비례해 가장 큰 항.
5. DFlash 때문에 줄어드는 max B: `ΔB_max = B_max,BL − B_max,DF`.

**예측되는 전이.** 요청 단위 prefill에서는 피크가 늘 마지막 행의 `prefill: target
forward`에 남는다. 하지만 그 피크를 지배하는 성분은 그 행의 prefill activation에서 다른
행들의 resident KV로 넘어간다. 8B는 B=2부터, 9B는 B≈5–8부터다. 이 전이를 성분 분해로
확인한다.

### 5.4 대역폭(bandwidth) 병목

**분석적 바이트/FLOP 모델.** step 하나, phase 하나 기준이다. 가중치는 **실제로 실행된
module만** 합산한다(hook으로 확인). 9B의 vision tower처럼 로드만 되고 실행되지 않는
가중치는 뺀다.

| 항 | Qwen3-8B | Qwen3.5-9B |
| --- | --- | --- |
| target 가중치 read | 약 14.1 GiB(embedding 표 제외, lm_head 포함) | 실행 module 합산 |
| target KV read | `B·L̄·144 KiB` | `B·L̄·32 KiB` |
| target KV write | `B·q·144 KiB` | `B·q·32 KiB` |
| GDN state read/write | — | 약 `B·96 MiB`(fp32, 24층, r+w) |
| draft 가중치 read | 1.95 GiB + target lm_head 한 번 더 | 2.41 GiB + target lm_head 한 번 더 |
| draft KV read | `B·L̄·20 KiB` | `B·(L̄·4 KiB + 5·4096·4 KiB)` |
| FLOPs (GEMM) | `2·P·B·q` | `2·P·B·q` |
| FLOPs (attention) | `4·B·q·L̄·H_q·d_h·L_attn` (36층) | 같은 식 (full-attn 8층) |

**Ridge.** A6000 사양은 768 GB/s, bf16 dense 154.8 TFLOPS이고 ridge는 약 200 FLOP/B다.

* weight GEMM의 강도는 약 `B·q` FLOP/B다. 그래서 verify(q=16)는 **B≈8–13에서
  compute-bound에 들어가고**, baseline(q=1)은 이번 sweep 범위에서 내내 memory-bound다.
* attention의 강도는 약 GQA group(4) × q다. verify는 64, baseline은 4로 둘 다
  memory-bound다.
* 사양값 대신 Phase 0의 microbenchmark로 **실측 ceiling**을 잡아 쓴다.

**측정.**

* phase별 event 시간과 분석적 바이트로 achieved GB/s와 TFLOPs, roofline 대비 비율을
  계산한다.
* 커널 수준은 `--profile-kernels K`로 **따로 진단 pass**를 돌린다. 본 측정과 분리해야
  profiler overhead가 throughput에 섞이지 않는다. torch.profiler로 steady step K개를
  잡고 커널을 이름으로 분류한다: GEMM / attention(backend 이름 포함) / copy·cat /
  GDN(torch fallback의 elementwise·cumsum·bmm) / norm·activation / sampling / 기타.
  카테고리별로 시간 비중과 achieved BW를 낸다.
* 하드웨어 카운터: **`ncu`가 없어서 DRAM throughput을 직접 잴 수는 없다.** `nsys`는
  있으므로 타임라인과 GPU idle을 교차 검증하는 데 쓴다. 보조로 NVML memory
  utilization(메모리 컨트롤러 busy %)을 100 ms 간격으로 샘플링한다.

**phase별 판정 규칙.** 기준값은 Phase 0 결과를 보고 조정한다.

| 판정 | 조건 |
| --- | --- |
| HBM-bound | achieved BW ≥ 실측 ceiling의 70% |
| compute-bound | achieved FLOPs ≥ 실측 ceiling의 60% |
| launch/CPU-bound | 둘 다 낮고 `T_cpu` 비중이 큼 |

### 5.5 정확성과 수락

* τ̄_B(행 평균과 pooled 모두), histogram, pre/post natural-EOS 분리.
* **batch invariance**: 같은 프롬프트의 B 출력과 B=1 출력이 얼마나 일치하는지, 그리고
  첫 불일치 위치의 분포. bf16 GEMM은 M(행 수)에 따라 다른 커널을 쓸 수 있다. 불일치가
  흔하면 B 간 τ 차이는 배치 효과가 아니라 수치 효과로 해석해야 한다.
* **losslessness**: 같은 B에서 DFlash 출력과 baseline 출력이 같은 비율.

## 6. 가설 (사전 예측)

**H1 (8B, 4k).** same-B speedup은 B가 커지면서 줄고, verify GEMM이 compute-bound에
들어가는 B≈8–16에서 급감한다. B=1의 1.82x에는 HF eager의 step당 CPU overhead(추정 수
ms)를 τ토큰에 나눠 갚는 효과도 들어 있다. B가 커지면 이 몫이 사라진다(`c_cpu`로 확인).

**H2 (8B, 16k–32k).** target KV read가 step 시간을 지배하므로 `c_verify`가 1 가까이
유지되고, 그래서 speedup이 배치에 덜 민감할 것이다(MagicDec 계열의 주장). 단 B=1에서
이미 speedup이 1보다 작았다(16k 0.90x, 32k 0.38x). 원인은 낮은 τ(2.50, 1.48)와 decode
토큰 부족이었으므로, N=256 고정에서 이것이 어떻게 달라지는지 먼저 본다. 8B drafter는
full attention이라 draft KV read(20 KiB/token, target의 14%)도 함께 커진다.

**H3 (9B).** KV가 32 KiB/token이라 KV-bound 영역에는 늦게 도달한다. step 시간은 GDN
torch fallback(fp32 chunk 알고리즘, q=16을 64로 pad)의 compute와 launch가 지배할
것이다. fla 커널을 쓰면 그림이 달라지므로 선택 ablation으로 둔다(§8 Phase 4).

**H4 (8B 용량).** max B는 target KV가 정한다. DFlash의 추가 resident(draft KV
20 KiB/token, 마지막 행의 tap hidden 40 KiB/token)는 이 격자에서 1–3 GiB 수준이라
max B를 한 단계 낮추지 못한다.

**H5 (9B 용량).** GDN record buffer를 없애고 나면 9B의 병목은 마지막 행의 fp32 prefill
activation(210 KiB/token)과 나머지 행의 resident가 된다. 32k에서 max B는 1에서
8(DFlash) / 16(baseline)으로 커진다. 이 격자에서 **DFlash만 들어가지 않는 유일한
점**은 32k × B=16이며, 이 점이 iso-memory 비교의 핵심 점이다.

## 7. 메모리 예측과 실행 격자

B=1 측정에서 가져온 계수(KiB/token/sequence)다. W는 target과 draft의 weight 합이다.

| | W (GiB) | target KV | GDN record (prefill) | tap hidden | prefill act | draft KV (steady) | 행당 고정 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-8b | 15.26 + 1.95 | 144 | — | 40 | ≈105 | 20 | — |
| qwen3.5-9b | 16.68 + 2.41 | 32 (full-attn 8층) | 384 | 64 | ≈210 | 4 (+ring buffer) | ≈0.13 GiB |

두 구조의 피크 식:

* **A. 현재 구조**(B행을 한꺼번에 prefill, 9B는 recording 유지):
  `W + B·S·(KV + record + tap + act)`
* **B. 제안 구조**(§4): `W + (B−1)·(S·(KV + draftKV) + 행당 고정) + S·(KV + tap + act)`
  와 decode steady 중 큰 쪽

A6000 48 GB에서 할당량 44 GiB 이하를 fit으로 본다. 식을 B=1에 넣으면 기존 측정과
일치한다(8B 4k 18.3 / 32k 26.2, 9B 4k 21.8 / 32k 40.7, 실측 18.34 / 26.23 / 21.82 /
40.54).

| max B (DFlash / baseline) | 8B · A | 8B · **B** | 9B · A | 9B · **B** |
| --- | --- | --- | --- | --- |
| 4k | 16 / 16 | **32 / 32** | 8 / 8 | **32 / 32** |
| 8k | 8 / 8 | **16 / 16** | 4 / 4 | **32 / 32** |
| 16k | 4 / 4 | **8 / 8** | 2 / 2 | **16** (32은 45.3 GiB로 tight) / **32** |
| 32k | 2 / 2 | **4 / 4** | 1 / 1 | **8 / 16** |

9B 수치는 GDN replay buffer(행당 약 +0.06 GiB)를 넣으면 조금 오른다. 그래도 16k × 32만
fit에서 빠지고 나머지 결론은 같다.

**실행 격자** (구조 B 기준, {1,2,4,8,16}과 fit하는 곳의 교집합에 32 추가):

| | 4k | 8k | 16k | 32k |
| --- | --- | --- | --- | --- |
| qwen3-8b | 1–16, 32 | 1–16 | 1–8 | 1–4 |
| qwen3.5-9b | 1–16, 32 | 1–16, 32 | 1–16 (32는 시도) | 1–8 (baseline은 16까지) |

여기에 경계 확인용으로 예측상 OOM인 점을 하나씩 추가한다(8B 32k B=8, 9B 32k DF B=16).

## 8. 실행 계획

**Phase 0: 검증.** GPU 1장, 반나절 정도.

| # | 검증 | 통과 기준 |
| --- | --- | --- |
| V0 | ceiling microbenchmark: device-to-device copy BW, bf16 GEMM TFLOPs(4096×4096×M, M=16…4096), mem-efficient SDPA의 KV read BW | §5.4 판정의 분모로 쓴다 |
| V1 | GDN 수정 | tiny 모델 rollback relative error ≤ 1e-6. 9B 4k 8 prompts에서 DFlash 출력 == baseline 출력. **수정 전후의 τ와 출력 길이를 비교해 기존 9B 기록이 받은 영향을 정량화** |
| V2 | ragged 정확성 | tiny 모델 fp32 CPU에서 B=4 ragged 결과가 행별 B=1 결과와 일치(1e-5). 실제 8B 4k B=4에서 행별 출력의 B=1 대비 일치율 |
| V3 | B=1 회귀 | 새 엔진 B=1과 기존 `dflash_generate`(8B 4k 8 prompts, 자연 종료)에서 `output_ids`와 τ가 같음. 시간은 `--kv-cache dynamic`으로 artifact 크기를 잰다 |
| V4 | 메모리 예측 | 8B 4k B=2, 4에서 예측과 실측이 ±5% 이내 |
| V5 | 계측 overhead | phase probe를 켜고 끈 tpot 차이 < 2% |
| V6 | baseline sanity | 새 엔진 baseline(B=1, 8)과 HF `generate`(static cache)의 throughput 비교. baseline이 인위적으로 느리지 않은지 확인 |

**Phase 1.** 8B 전체 격자(GPU 0).

**Phase 2.** 9B 전체 격자(GPU 1). V1을 통과한 뒤에 시작한다.

**Phase 3.** 진단 pass(`--profile-kernels 8`). 두 모델 × {4k, 32k} × {1, max_B/2, max_B}
× 두 모드.

**Phase 4 (선택).**

* prefill batched vs sequential (8B 4k, B≤8)
* kv-cache dynamic vs static (B=1, 8B 32k)
* 9B에 `kernels`/fla 설치 ablation. 결과는 별도 디렉터리에 두고 비교하지 말라고 표기한다.

**GPU.** 이 설계를 쓰는 시점(2026-09-10)에 0, 1번은 idle이었고 2, 3번은 다른 프로세스가
쓰고 있었다(각 약 30 GB, 100%). 기존 스크립트처럼 idle인지 확인한 뒤에만 잡는다.

**소요 시간(기존 ttft와 tpot으로 추정).** 8B 약 2시간, 9B 약 3시간, 병렬로 약 3시간.
절반 이상이 prefill이다. 요청마다 prefill을 하므로 B 5개 × 모드 2 × 32요청만큼 반복된다.

## 9. 기록 형식과 시각화

```jsonc
{
  "git_commit": "…", "model_name": "qwen3-8b", "context_length": 16384, "batch_size": 8,
  "status": "ok",                     // "oom" | "skipped_predicted_oom"
  "oom_phase": null,
  "fixed_output_tokens": 256, "num_prompts": 32,
  "prefill_mode": "sequential", "kv_cache": "static",
  "sdpa_backend": {"prefill": "flash", "verify": "efficient"},
  "predicted_peak_gb": {"dflash": 39.7, "baseline": 36.9},
  "summary": {"dflash": { … }, "baseline": { … }},
  "speedup": {"same_b_full_occupancy": …, "same_b_makespan": …},  // vs-B1, iso-memory는 분석 단계에서 교차 계산
  "batches": [{"rows": [3, 17, …],
               "steps": [{"wall_s": …, "verify_s": …, "draft_s": …, "ctx_s": …,
                          "accept_s": …, "active": 8, "tokens": 29}, …]}],
  "rows": [{"prompt": 3, "acceptance_lengths": […], "natural_eos_at": 37,
            "matches_b1": true, "matches_baseline": true}]
}
```

`summary`의 주요 키: `decode_tok_s_full_occupancy`, `decode_tok_s_makespan`, `e2e_tok_s`,
`mean_ttft_s`, `drafter_prefill_s`, `tpot_s`, `mean_acceptance_length`(all / pre-EOS /
post-EOS), step 분해의 평균과 `c_*` 비율, `phase_memory`(계수 포함), `peak_gb`,
`peak_phase`, 피크 성분 분해, phase별 roofline(bytes, flops, achieved_gbps,
achieved_tflops, bound).

`visualization_batch/`의 그림:

| # | 그림 |
| --- | --- |
| F1 | throughput vs B. S별로, 두 모드, full-occupancy와 makespan |
| F2 | same-B speedup vs B(S별 선)와 vs-B1 speedup |
| F3 | step 시간 누적 막대(verify / draft / ctx / accept / cpu) vs B, S별, DF vs BL, step당 토큰 병기 |
| F4 | `c_verify(B,S)` heatmap. compute-bound에 들어가는 위치 |
| F5 | 피크 메모리 vs B. 피크 시점 성분별 누적, 예측선, OOM 표시 |
| F6 | throughput–피크 메모리 Pareto(iso-memory) |
| F7 | roofline scatter. phase별 (arithmetic intensity, achieved perf), 색은 B |
| F8 | 커널 카테고리 비중 vs B(진단 pass) |
| F9 | τ̄ vs B와 batch invariance / losslessness 일치율 |
| F10 | straggler 효과. full-occupancy vs makespan, step에 따른 active row 수 |

## 10. 함정

1. **새 엔진의 B=1 시간은 기존 record와 다르다.** static cache가 cat 복사를 없애기
   때문이다. 기존 기록과는 τ와 메모리만 비교하고, 시간은 `--kv-cache dynamic` 교량
   측정을 통해서만 연결한다.
2. **기존 9B DFlash 기록은 GDN rollback 버그 아래에서 측정되었다.** 새 결과와 τ나
   speedup을 비교하지 않는다. 메모리는 비교해도 된다.
3. **EOS를 억제한 post-EOS 구간은 자연스러운 분포가 아니다.** τ는 pre-EOS만 따로
   보고한다. 두 값이 크게 다르면 헤드라인에 pre-EOS 기반 추정치를 함께 적는다.
4. **정적 배치의 makespan throughput에는 straggler 비용이 섞여 있다.** 배치 step 비용을
   비교할 때는 full-occupancy 값을 쓴다.
5. **sharded 점은 없다.** 들어가지 않는 점은 OOM으로 기록할 뿐 sharding으로 채우지
   않는다. sharded 점과는 시간을 비교하지 않는다는 규칙 때문이다.
6. **profiler 진단 pass에서 잰 시간은 throughput 표에 쓰지 않는다.**
7. **분석적 바이트 모델에는 실제로 실행된 module만 넣는다.**
8. **수치 비불변성.** 배치에 따라 greedy 출력이 달라질 수 있다. B 간 τ 차이는 §5.5의
   일치율과 함께 해석한다.

## 부록 B. 구현하면서 설계와 달라진 점

`dflash/batch.py`, `dflash/batch_bench.py`, `dflash/batch_report.py`,
`queue/run_batch_sweep.sh`, `queue/compare_old_path.py`,
`tests/test_batch_engine.py`로 구현했다. 설계와 다른 결정은 모두 측정에
근거한다.

1. **Attention은 SDPA mask가 아니라 kernel 세 종류 중 실측으로 고른다.**
   §4.2의 "행별 bool mask + mem-efficient"는 A6000에서 느렸다. 32k 단일 토큰
   decode에서 flash보다 4배 느렸고, 새 엔진의 baseline이 기존 경로보다 4 ms
   느려지는 원인이었다.
   - 후보는 셋이다: dense flash(행 길이가 모두 같을 때, split-KV), FA2 varlen +
     `seqused_k`(ragged 행), GQA를 접은 masked mem-efficient(q ≤ 64).
   - shape마다 첫 호출에서 셋을 재고 가장 빠른 것을 쓴다. 선택 결과는 레코드의
     `summary.*.attention_kernels`에 남는다.
   - GQA를 query 길이로 접은 varlen은 4~7배 느려서 쓰지 않는다.
2. **KV 저장 layout은 seq-major `(B, cap, H_kv, d)`다.** varlen이 복사 없이
   `(B·cap, H_kv, d)`로 볼 수 있어야 하기 때문이다. 쓰기는 한 번의 op로 한다:
   uniform이면 slice `copy_`, ragged면 `index_copy_`.
3. **GDN decode chunk는 16이다.** torch fallback은 16토큰 verify를 64로 padding한
   뒤 Python loop를 63번 돈다. chunk 16이면 verify가 180 ms → 93 ms로 줄고,
   결과는 수학적으로 동일하다(CPU fp32 테스트 통과). replay는 host sync 뒤에
   그 step의 최대 kept 길이까지만 돈다.
4. **9B draft의 sliding layer는 ring buffer 대신 full-length storage에 둔다.**
   flash의 `window_size_left`가 window 밖 block을 건너뛰므로 읽기 비용은 window
   크기다. 메모리는 24 KiB/token(draft 6층분)이라 §7의 4 KiB/token보다 크다.
   실측에서 이 결정이 경계를 바꾼 곳은 둘이다: 9B 16k×32와 32k×16에서
   baseline은 들어가고 DFlash만 cache 할당에서 OOM이 났다. 9B는 target KV가
   32 KiB/token으로 작아서, draft KV가 target KV의 75%에 이른다. ring buffer로
   바꾸면 16k×32에서 약 7.8 GB가 줄어 경계선 근처로 들어온다.
5. **step당 host sync는 1회다.** 행 길이와 active 여부는 host에서 추적한다.
   진단 결과 추가 sync 두 개(`lengths.max()`, `torch.tensor(active)`)가 CPU/GPU
   overlap을 깨서 9B q=1 step을 35 → 47 ms로 늘리고 있었다.
6. **프롬프트는 정확히 S 토큰인 것만 쓴다.** 풀을 16개 넉넉히 만든 뒤 걸러낸다.
   baseline은 모든 행이 lockstep으로 진행하므로, 길이가 같아야 dense kernel을
   쓸 수 있다.

**스윕 도중 발견해 고친 버그 두 개** (2026-09-10). 둘 다 9B에만 걸렸고, 영향받은
레코드는 지우지 않고 옮겨 두었다. 해당 점은 고친 코드로 다시 쟀다.

- **cache 누수: `TargetCache` ↔ `_GdnLayerView` 순환 참조.**
  - view가 cache를 강하게 잡고 있었다. 그래서 끝난 batch의 KV와 GDN state가
    Python cyclic GC가 돌 때까지 GPU에 남았다.
  - 증상은 9B 피크가 요청마다 한 요청분(32k에서 1.06 GB)씩 오르다 GC 때 떨어지는
    것이다. 9B 32k B=1 DFlash는 이 누적으로 prefill에서 OOM이 났다. 8B에는 view가
    없어서 영향이 없었다.
  - 고친 뒤 피크는 요청마다 같다(32k B=1 baseline 26.93 GB). 전에는 28.0 → 37.5 GB였다.
  - 수정은 view가 `weakref.proxy`로 cache를 잡게 한 것이다. 테스트는
    `test_caches_freed_by_refcount`다. 옛 레코드는 `record_batch/stale_gc_leak/`에 있다.
- **draft mask 공유: `_masked`의 mask cache key에 `causal`과 `window`가 없었다.**
  - 한 step의 `meta`는 모든 layer가 공유한다. 9B draft는 full layer(non-causal)와
    sliding layer(causal, window 4096)가 같은 shape이다. 그래서 둘 다 masked
    kernel로 autotune된 step에서는 먼저 만든 mask를 다른 종류의 layer가 썼다.
  - 이 조합이 된 곳은 9B 4k B≥4와 8k B≥8뿐이다.
  - target 출력은 verify가 지키므로 lossless였고, 떨어진 것은 수락뿐이다. 이 구간의
    τ가 4.2 → 3.5로 떨어졌고, 대부분 post-EOS 구간에서였다.
  - CPU 테스트는 target 출력만 비교해서 이를 잡지 못했다.
  - 수정은 key에 두 값을 넣은 것이다. 테스트는 `test_attend_mask_per_layer_kind`로,
    draft 모델의 `_attention_mask`를 기준으로 비교한다. 옛 레코드는
    `record_batch/stale_mask_collision/`에 있다.

**B=1 교량 측정** (`record_batch/compare_old_*.json`, 4k, 8 prompts, EOS 무시,
256 tokens):

| | old baseline | new baseline | old DFlash | new DFlash | old speedup | new speedup | old DF == old BL | new DF == new BL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-8b (ms/token) | 30.9 | 29.6 | 12.1 | 9.4 | 2.56 | 3.15 | 6/8 | 5/8 |
| qwen3.5-9b (ms/token)¹ | 38.0 | 38.5 | 49.5 | 44.2 | 0.77 | 0.87 | **0/8** | 6/8 |

¹ 9B의 new 수치는 attention autotune 이전 코드로 측정했다.

- 8B는 새 baseline이 기존보다 빠르고 출력도 8/8 동일하다. 따라서 speedup
  상승분(2.56 → 3.15)은 기존 경로가 q=16 verify에만 부과하던 artifact가
  사라진 효과다. 그 artifact는 매 step의 KV cat 복사, 그리고 mask가 있을 때
  `repeat_kv`로 생기는 GQA KV 복사다.
- 9B는 기존 경로에서 DFlash 출력이 baseline과 한 번도 같지 않았다(0/8). §2.2의
  rollback 버그가 실제 모델 출력을 바꿨다는 직접 증거다.

## 부록 C. fla kernel 재측정 (2026-09-12)

`fla` 0.5.2를 설치하고 9B 격자를 다시 돌려 `record_batch_fla/`에 기록했다. 8B는 GDN이
없어 fla를 호출하지 않으므로 다시 돌리지 않았다.

1. **설치는 overlay venv로 했다.** `python -m venv --system-site-packages`로 만든
   `~/venvs/dflash-fla`에 `--no-deps`로 `fla-core`와 `flash-linear-attention`을 넣었다.
   기존 env는 fallback 기준선으로 남겨 두었다. 스윕 스크립트는 `REC`/`LOGDIR`/`PY`를
   환경변수로 받는다.
2. **`fla-core`만으로는 조용히 fallback으로 돈다.** transformers는
   `fla.ops.gated_delta_rule`을 getattr 체인으로 찾는데, fla-core만 있으면 `fla.ops`가
   import되지 않아 `None`이 나오고 torch 함수가 그대로 남는다. 경고도 에러도 없다.
   그래서 레코드에 `gdn_kernels`(바인딩된 구현 이름과 fla 버전)를 적게 했다.
3. **엔진은 그대로 동작한다.** capture는 decorator가 만든 wrapper를 감싸므로 replay도
   fla를 탄다. 엔진이 넘기는 `chunk_size=16`은 signature에 없어 조용히 버려진다(fla는
   내부적으로 64를 쓴다). g=beta=0 masking은 fla에서도 정확했고(kept=0 행이 변하지 않음),
   측정된 τ는 모든 점에서 ±0.1 안이었다.
4. **CPU 테스트는 fla env에서 깨진다.** Triton이 CPU tensor를 받는다. 기존 env에서 돌린다.
5. **`causal_conv1d`와 `kernels`는 설치하지 않았다.** conv는 여전히 torch 경로다.

## 부록 A. GDN rollback 재현 코드

`CUDA_VISIBLE_DEVICES="" python`으로 실행한다. 레포 루트에서 실행해야 `dflash`를
import할 수 있다.

```python
import torch
from transformers.models.qwen3_5 import configuration_qwen3_5 as C, modeling_qwen3_5 as M
from dflash.model import _make_cache, _crop_to

torch.manual_seed(0)
cfg = C.Qwen3_5TextConfig(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
    linear_value_head_dim=16, linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "full_attention"], max_position_embeddings=512,
)
model = M.Qwen3_5TextModel(cfg).eval().float()
ids = torch.randint(0, 128, (1, 48))
P, V, KEEP = 32, 8, 3

@torch.no_grad()
def run(steps):
    cache = _make_cache(cfg)
    for a, b, crop in steps:
        out = model(input_ids=ids[:, a:b], position_ids=torch.arange(a, b)[None],
                    past_key_values=cache, use_cache=True).last_hidden_state
        if crop is not None:
            _crop_to(cache, crop)
    return out[:, -1]

ref      = run([(0, P + KEEP, P + KEEP), (P + KEEP, P + KEEP + 1, None)])
chunked  = run([(0, P, P), (P, P + KEEP, P + KEEP), (P + KEEP, P + KEEP + 1, None)])
rollback = run([(0, P, P), (P, P + V, P + KEEP), (P + KEEP, P + KEEP + 1, None)])
rel = lambda x: ((x - ref).norm() / ref.norm()).item()
print(rel(chunked), rel(rollback))   # 7.3e-08, 6.7e-03
```
