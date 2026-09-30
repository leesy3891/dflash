# DFlash의 long-context 병목: 아키텍처별로, DFlash의 어느 단계에서 생기는가

- 작성일: 2026-09-30
- 측정 체계: `dflash.arch`의 measurement protocol v2. 측정값의 정의는 `MEASUREMENT_OUTPUTS.md`, 결함 수정 이력은 `MEASUREMENT_AUDIT.md`에 있다.
- 공통 조건
  - 3 GPU(RTX A6000)에 layer 단위로 샤딩, B=1, greedy, block 16, 출력 256 토큰(EOS 무시).
  - 프롬프트: LongBench 2wikimqa, source_index 66의 같은 문서를 모든 S에서 사용한다(8k 이상은 같은 과제의 passage를 덧붙여 확장).
  - GDN 커널 백엔드: `causal_conv1d+fla`.
- 모든 수치 뒤의 `[Rx]`는 그 값을 확인할 수 있는 record다(1절 표). CSV는 각 record 옆의 `csv/<record 이름>/`에 있다. trace 분해 결과는 `traces/<stamp>/*.breakdown.json`에 있으며 `python -m dflash.arch trace-breakdown`으로 다시 만들 수 있다.

---

## 0. 요약

| 연구 질문 | 답 |
|---|---|
| **Q1.** target attention KV가 작아질 때 DFlash 상태의 비중 | 비중은 커진다. 다만 커지는 주체는 draft KV가 아니다. **GDN rollback용 conv recording buffer**(64k에서 9B 25.8 GB, 35B 32.2 GB로 attention KV의 12배, 24배)와 **프롬프트 길이 context feature**(64k에서 9B 4.30 GB, attention KV의 2배)다. 둘 다 prefill 끝에서 첫 verify 사이에만 존재하는 일시적 항목이며, 이 구간이 DFlash의 메모리 피크를 결정한다. steady 상태의 draft KV 비율은 drafter 구조로 정해진다. 8B는 0.139(= 5/36층), 9B와 35B는 0.16–0.26이다(sliding window 층). |
| **Q2.** 35B-A3B verify의 층별 비용 | 4k에서 verify 1회의 장치 시간은 54 ms다. 그중 expert GEMM이 30.6 ms(56%)이고 route 1.3 ms, dispatch 2.1 ms, GDN 8.0 ms, attention 6.5 ms, lm_head 1.4 ms다. 그러나 벽시계는 183 ms로 **장치 시간은 30%뿐이고 나머지는 호스트의 커널 launch**다(verify 1회에 9,239개, 그중 5,808개가 expert 루프). 64k에서는 attention이 83 ms로 커져 최대 항목이 된다. |
| **Q3.** S와 block 폭에 따른 단계별 변화 | 모든 모델에서 decode 시간의 81–95%는 **verify**다. S가 커질 때 verify를 키우는 것은 full-attention 층이다. verify는 bool 마스크를 materialize해서 mem-efficient 커널과 복사 연산을 쓰고, 그 비용이 S에 비례한다. first draft(context 투영)는 S에 선형이지만 decode의 0.3–8.8%에 그친다. 병목의 성격은 모델마다 다르다. **8B는 GPU-bound(attention)**, **9B는 짧은 S에서 일부 host-bound였다가 긴 S에서 GPU-bound**, **35B는 전 구간 host-bound**다. block 폭을 넓히면 35B에서는 이득이 크고(b4 65 ms에서 b16 35 ms), 9B와 8B에서는 차이가 거의 없다. |

아키텍처별 long-context 병목과 그 병목이 생기는 DFlash 단계는 6절 표에 모았다.

---

## 1. 근거 record

| ID | 파일 | 내용 |
|---|---|---|
| **R8** | `record_arch_main/qwen3-8b/sweep_causal_conv1d+fla_3gpu_20260930-113850.json` | Qwen3-8B, S=4k–64k와 block {4,8,16}@32k, 패스 perf/memory/trace |
| **R9** | `record_arch_main/qwen3.5-9b/sweep_causal_conv1d+fla_3gpu_20260930-103640.json` | Qwen3.5-9B, 같은 조건 |
| **R35** | `record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_3gpu_20260930-023904.json` | Qwen3.5-35B-A3B, S=4k–64k, 패스 perf/memory/moe/trace |
| **R35B** | `record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_3gpu_20260930-091225.json` | 35B-A3B, block {4,8,16}@32k, 패스 perf/memory/moe/trace |
| **R35X** | `record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_3gpu_20260930-081655.json` | 35B-A3B, 64k, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **R35-2** | `record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_2gpu_20260929-192934.json` | 35B-A3B, 2 GPU(용량 한계 확인용) |
| **T8, T9, T35** | `record_arch_main/{qwen3-8b,qwen3.5-9b,qwen3.5-35b-a3b}/traces/{20260930-113850, 20260930-103640, 20260930-023904}/` | R8, R9, R35의 trace 파일(decode 앞쪽 32 스텝)과 `*.breakdown.json` |

위 record의 모든 시간 측정 행은 `profiler_used_before=False`이고, AR 행은 `drafter_resident=False`다(확인함). 배치는 R8 12/12/12층, R9 10/11/11층, R35 13/13/14층이다.

아키텍처:

| | target 층 | FFN | drafter(`load.draft_report`) |
|---|---|---|---|
| Qwen3-8B | full attention 36 | dense | 5층 모두 full attention, 2.10 GB |
| Qwen3.5-9B | GDN 24 + full attention 8 | dense | sliding 5 + full 1, 2.58 GB |
| Qwen3.5-35B-A3B | GDN 30 + full attention 10 | MoE(256 expert, top-8) + shared | sliding 5 + full 1, 0.77 GB |

---

## 2. 실험 결과 정리

### 2.1 지연 시간과 speedup (perf 패스, block 16)

| 모델 | S | AR native | DFlash TPOT | acceptance | 확정 토큰/스텝 | speedup | verify 1회 | verify ÷ AR 스텝 | 근거 |
|---|---|---|---|---|---|---|---|---|---|
| 8B | 4k | 37.7 ms | 19.25 ms | 0.130 | 2.90 | 1.96× | 47.8 ms | 1.27 | R8 |
| 8B | 16k | 39.9 | 50.63 | 0.085 | 2.24 | 0.79× | 103.5 | 2.59 | R8 |
| 8B | 64k | 68.5 | 253.46 | 0.019 | 1.27 | **0.27×** | 306.5 | **4.48** | R8 |
| 9B | 4k | 42.0 | 8.14 | 0.534 | 8.79 | 5.16× | 59.5 | 1.41 | R9 |
| 9B | 16k | 41.9 | 7.60 | 0.604 | 9.81 | 5.52× | 60.1 | 1.43 | R9 |
| 9B | 32k | 41.5 | 23.83 | 0.170 | 3.49 | 1.74× | 69.0 | 1.66 | R9 |
| 9B | 64k | 41.9 | 12.72 | 0.591 | 9.81 | 3.29× | 100.9 | 2.41 | R9 |
| 35B | 4k | 131.7 | 28.71 | 0.402 | 6.71 | 4.59× | 183.4 | 1.39 | R35 |
| 35B | 16k | 131.4 | 33.76 | 0.349 | 6.07 | 3.89× | 195.2 | 1.49 | R35 |
| 35B | 64k | 132.2 | 38.35 | 0.417 | 7.08 | 3.45× | 257.3 | 1.95 | R35 |

8k와 32k 값(9B 32k 제외)은 R8, R9, R35와 각 `csv/.../speedup.csv`에 있다.

- speedup은 acceptance와 verify 비용의 곱으로 정해진다. acceptance는 S를 따라 단조롭게 변하지 않는다(9B: 0.53, 0.24, 0.60, 0.17, 0.59). 같은 문서라도 S가 바뀌면 모델이 생성하는 텍스트(추론 형식)가 바뀌기 때문이다[R9 `output_token_ids`]. 그래서 **S의 효과는 acceptance가 아니라 verify 비용(verify ÷ AR 스텝)으로 읽어야 한다.** 이 비율은 세 모델 모두 S에 따라 단조롭게 증가한다.
- 8B는 16k부터 DFlash가 AR보다 느리다. acceptance가 낮은(0.13에서 0.02) 상태에서 verify 비용이 4.48배까지 커지기 때문이다.
- 출력은 세 모델 모두 추론 텍스트다. EOS는 8B의 4k–16k에서만 190–240번째 토큰에 나오고, 그 외에는 256토큰 안에 없다. 그러므로 acceptance가 EOS 이후 텍스트에서 측정된 값은 아니다[R8, R9, R35 `ar.native.samples[0].output_token_ids`].

### 2.2 decode 시간의 단계별 구성 (perf 패스 CUDA event 타이머, decode 대비 비율)

| 모델 | S | verify | first draft | steady draft | draft logits | 미귀속 | 근거 |
|---|---|---|---|---|---|---|---|
| 8B | 4k | 85.7% | 0.3% | 9.5% | 3.6% | 0.9% | R8 |
| 8B | 64k | **94.8%** | 0.3% | 4.1% | 0.6% | 0.2% | R8 |
| 9B | 4k | 83.1% | 1.1% | 10.6% | 4.5% | 0.6% | R9 |
| 9B | 64k | 80.9% | **8.8%** | 7.3% | 2.6% | 0.3% | R9 |
| 35B | 4k | 95.2% | 0.2% | 3.3% | 1.0% | 0.3% | R35 |
| 35B | 64k | 94.7% | 1.5% | 2.9% | 0.7% | 0.2% | R35 |

context feature 구축(decode)은 전 조건에서 0.06% 이하다.

단계별 절대 시간(ms):

| 모델 | first draft 4k → 64k | steady draft 1회 4k → 64k | draft logits/스텝 | TTFT 증가(DFlash − AR) 64k | 근거 |
|---|---|---|---|---|---|
| 8B | 15.7 → 187.7 | 5.33 → 13.27 | 2.00 | 226 ms | R8 |
| 9B | 23.7 → 286.5 | 7.87 → 9.49 | 3.24 | 314 ms | R9 |
| 35B | 11.6 → 148.6 | 6.53 → 8.04 | 1.87 | 150 ms | R35 |

### 2.3 메모리 (memory 패스의 구성 요소, GB, 64k)

| 모델 | target attn KV | GDN 상태 | conv recording buffer (prefill 끝) | context feature (prefill 끝) | draft KV (first draft → steady) | 선택된 hidden (steady) | drafter 가중치 | 동시 피크(위치) | 근거 |
|---|---|---|---|---|---|---|---|---|---|
| 8B | 9.66 | 0 | 0 | 2.68 | 1.343 → 1.347 | 0.001 | 2.10 | 37.9 (target layer 35) | R8 |
| 9B | 2.15 | 0.052 | **25.77** | **4.30** | 1.611 → 0.353 | 0.001 | 2.58 | 62.9 (target layer 30) | R9 |
| 35B | 1.34 | 0.065 | **32.21** | 2.15 | 1.611 → 0.353 | 0.001 | 0.77 | 116.8 (target layer 39) | R35 |

AR 대비 메모리 오버헤드(장치별 최댓값 합의 차, perf 패스, 64k):

| 모델 | native AR 대비 | recording AR 대비 | 근거 |
|---|---|---|---|
| 8B | 3.17 GB | 3.17 GB | R8 |
| 9B | 29.43 GB | 3.66 GB | R9 |
| 35B | 33.52 GB | 32k에서 1.04 GB (64k recording AR은 OOM) | R35 |

---

## 3. Q1. target attention KV가 작아질 때 DFlash 상태의 비중

**근거:** R8, R9, R35의 `csv/.../component.csv`(시점: `prefill_end`, `first_draft`, `steady_decode`)와 `speedup.csv`.

1. **target attention KV의 크기.** 64k에서 8B 9.66 GB, 9B 2.15 GB, 35B 1.34 GB다[R8, R9, R35]. 하이브리드 모델은 full-attention 층이 8개, 10개뿐이어서 8B의 1/4.5, 1/7.2 수준이다. GDN 상태는 S와 무관하게 52 MB, 65 MB로 고정이다.

2. **draft KV의 비율은 drafter 구조가 정한다.**
   - 8B: draft KV ÷ attention KV가 모든 S와 시점에서 **0.139**로 일정하다. drafter의 5개 층이 모두 full attention이고 KV 형상이 target과 같으므로, 5/36 = 0.139다. 그래서 steady 상태에서도 draft KV가 S에 비례해 64k에서 1.35 GB가 된다[R8].
   - 9B: first draft 시점의 비율은 **0.75**(= 6/8)다. steady에서는 0.16으로 떨어진다. drafter 6층 중 5층이 sliding window라서, 첫 호출 뒤 crop되면 window만큼만 남기 때문이다. steady의 draft KV는 0.353 GB다[R9].
   - 35B: first draft 시점 1.20, steady 0.26이다[R35]. drafter가 9B와 같은 형상의 KV를 쓰는데 target attention KV가 더 작아서, 첫 호출 시점에는 **draft KV가 target attention KV보다 크다.**
   - 결론: attention KV가 작은 하이브리드 target일수록 draft KV의 **상대** 비중은 커진다. steady 상태의 절대 크기는 drafter의 sliding window 덕분에 제한된다(64k에서 0.35 GB).

3. **선택된 hidden state와 context feature.**
   - steady 상태에서 선택된 hidden은 약 1 MB다. 수락된 토큰만 전달되므로 무시할 수 있다[R8, R9, R35].
   - 그러나 **prefill이 만드는 프롬프트 길이의 context feature**는 64k에서 8B 2.68 GB, 9B 4.30 GB, 35B 2.15 GB다. 9B에서는 target attention KV의 2.0배, 35B에서는 1.6배다. 이 텐서는 prefill 끝부터 첫 draft 호출과 첫 verify까지 살아 있다가 첫 context feature를 재구축할 때 해제된다[R9, R35 `component.csv`의 `prefill_end`, `first_draft` 행].
   - context projection(첫 draft 호출이 S개 context를 draft KV로 투영하는 비용)은 64k에서 8B 188 ms, 9B 287 ms, 35B 149 ms다[R8, R9, R35 `draft_forward_first_s`].

4. **가장 큰 DFlash 기인 항목은 GDN rollback용 conv recording buffer다.**
   - DFlash는 거절된 블록을 되돌리기 위해 target cache의 past recording을 켠다. 이때 GDN 층의 conv 상태가 프롬프트 전체 길이로 남는다.
   - 64k prefill 끝에서 9B 25.77 GB, 35B 32.21 GB로 attention KV의 12배, 24배다. 첫 crop 이후에는 6–8 MB로 줄어든다[R9, R35].
   - DFlash의 native AR 대비 메모리 오버헤드(9B 29.4 GB, 35B 33.5 GB)는 대부분 이 버퍼에서 온다. recording AR 대비로 보면 3.7 GB, 1.0 GB에 그친다[R9, R35 `speedup.csv`].
   - 동시 피크는 세 모델 모두 prefill의 마지막 target 층에서 난다[`peak_site`].

**결론.** target attention KV가 작아지면 DFlash가 추가하는 상태의 비중은 확실히 커진다(8B: native 대비 +3.2 GB = attention KV의 33%. 9B: +29.4 GB = 13.7배). 그러나 원인은 draft KV나 선택된 hidden이 아니다. **GDN rollback 버퍼와 프롬프트 길이 context feature라는 두 prefill 단계 항목**이 원인이다. 두 항목 모두 상주 상태가 아니라 prefill 끝에서 첫 verify 사이에만 존재한다.

---

## 4. Q2. Qwen3.5-35B-A3B의 층별 verify 비용

**근거:** R35, T35의 `*__dflash.decode_target_verify.breakdown.json`과 `*__ar.decode_ar_step.breakdown.json`(S=4k, 64k), R35의 `passes.moe`와 `ar_moe`(`csv/.../moe_routing.csv`).

### 4.1 verify 1회의 구성 (장치 시간, ms/회, trace 32 스텝 평균)

| 구성 | S=4k verify | S=64k verify | S=4k AR 스텝 | 커널 종류 |
|---|---|---|---|---|
| **route** (`mlp.gate`, top-8 선택) | 1.26 | 1.26 | 1.01 | index/scatter 0.58, GEMM 0.24, 기타 |
| **dispatch** (expert 루프 안의 gather/scatter/elementwise) | 2.11 | 2.11 | 1.86 | elementwise 1.00, index 0.70, 기타 0.41 |
| **expert GEMM** (routed experts) | **30.58** | **31.86** | 4.23 | GEMM |
| shared expert | 1.03 | 1.03 | 0.77 | |
| GDN mixer (30층) | 8.01 | 7.99 | 5.05 | GEMM 4.27, GDN/conv 1.88, elementwise 1.77 |
| full attention mixer (10층) | 6.51 | **83.23** | 2.18 → 7.53 (64k) | mem-efficient 2.92 → 45.01, elementwise 2.54 → 37.16 |
| lm_head | 1.40 | 1.40 | 1.36 | |
| 모듈 밖(norm, residual 등) | 2.05 | 2.05 | 1.73 | |
| **합계(장치 시간)** | **54.4** | **132.7** | 19.1 | |
| **벽시계(perf)** | **183.4** | **257.3** | 131.7 | [R35] |
| 커널 launch 수 | 9,239 | 9,453 | 4,464 | [R35 `trace` 패스] |

층별 평균(층 전체 = mixer + MoE FFN, [R35 `passes.trace.median.analysis.phases["decode: target verify"].layer_s`]):

| 층 유형 | 4k verify | 64k verify | 4k AR | 64k AR |
|---|---|---|---|---|
| full attention + MoE (10층) | 1.55 ms | **9.24 ms** | 0.43 | 0.96 |
| GDN + MoE (30층) | 1.16 ms | 1.19 ms | 0.39 | 0.38 |

### 4.2 해석

- **expert GEMM이 verify 장치 시간의 최대 항목이다(4k 기준 56%).**
  - verify의 16토큰은 층마다 평균 48.5개의 expert를 건드린다(최대 100개, 256개 중). AR은 8개다. 부하 불균형(최대 ÷ 평균)은 4.04다[R35 `moe_routing.csv`].
  - expert 루프는 hit expert마다 GEMM을 launch하므로, expert GEMM이 AR의 7.2배, 루프 커널 수가 3.9배(1,480 → 5,808)가 된다[T35 breakdown].
  - route(1.26 ms)와 dispatch(2.11 ms)는 작고 S와 무관하다.
- **그러나 verify의 벽시계는 장치 시간의 3.4배다.**
  - 4k에서 verify 벽시계 183 ms 중 GPU가 일한 시간은 54 ms(30%)이고, 64k에서도 52%다. AR은 14–18%다[R35 perf와 trace].
  - 나머지 시간은 호스트가 9,000개가 넘는 커널을 launch하는 데 쓰인다(HF 참조 MoE의 expert별 Python 루프).
  - 그래서 35B verify ÷ AR 스텝의 벽시계 비율(1.39)은 장치 시간 비율(2.85)보다 launch 수 비율(2.07)에 가깝다.
- **GDN은 S와 무관하다**(8.0 ms/verify, 층당 약 0.27 ms). recurrent 상태 크기가 고정이기 때문이다.
- **attention은 S에 비례해 커지며, 64k에서 verify 장치 시간의 63%다.** verify에는 bool 마스크 `[1,1,16,S]`가 전달된다[R35 `attention_calls.csv`]. 그래서 SDPA가 flash 대신 mem-efficient 커널을 쓰고, elementwise 복사(37 ms)가 더해진다. full-attention 층 하나가 verify에서 9.24 ms로 GDN 층의 7.8배다.
- **draft와 target 경계**(스텝당)[R35 trace]
  - draft forward: `cuda:0`에서 4.6 ms(4k)에서 6.2 ms(64k), 약 495 커널
  - draft logits: target의 lm_head(마지막 장치)에서 1.84 ms, 커널 3개
  - 장치 간 P2P 복사: `cuda:0` 0.17–0.63 ms, `cuda:2` 0.84 ms
  - 경계 비용은 verify의 2% 미만이다.

---

## 5. Q3. S와 block 폭에 따른 단계별 변화

### 5.1 S 증가: 어느 단계가 어떤 자원에 묶이는가

**근거:** R8, R9, R35(perf, trace), T8, T9, T35 breakdown.

장치 시간 ÷ 벽시계(trace의 장치 시간 합 ÷ perf 벽시계). 낮을수록 host-bound다.

| 모델 | AR 스텝 4k → 64k | verify 4k → 64k | 해석 |
|---|---|---|---|
| 8B | 77% → 97% | 93% → 99% | GPU-bound |
| 9B | 66% → 86% | 57% → 98% | 짧은 S에서는 일부 host-bound, 긴 S에서는 GPU-bound |
| 35B | 14% → 18% | 30% → 52% | host-bound(커널 launch) |

GPU-bound 구간이 대역폭에 묶였는지 연산에 묶였는지는 하드웨어 카운터 없이 판정할 수 없다(ncu를 쓸 수 없음, `MEASUREMENT_AUDIT.md` F절). 아래는 **기록된 값으로 계산한 추정**이다. A6000의 공칭 대역폭은 768 GB/s다.

| 커널(64k, 8B) | 시간 | 읽는 양 | 실효 대역폭 | 근거 |
|---|---|---|---|---|
| AR flash(splitkv), 36층 | 13.31 ms | KV 9.66 GB | ≈ 726 GB/s(공칭의 약 95%) | R8 `trace_kernel.csv`, `component.csv` |
| AR KV `torch.cat`(72회) | 27.01 ms | KV 읽기와 쓰기 약 19.3 GB | ≈ 715 GB/s | 같음 |
| verify mem-efficient, 36층 | 119.34 ms | KV 9.66 GB | ≈ 81 GB/s | 같음 |
| verify elementwise 복사(`direct_copy`, 103회) | 110.26 ms | 미상 | — | 같음 |
| AR과 verify의 dense FFN | 16.0 ms | MLP 가중치 약 10.9 GB(36층 × 3 × 4096 × 12288 × 2 B) | ≈ 680 GB/s | T8 breakdown |

- **AR 스텝**은 긴 S에서 대역폭에 묶인다. KV를 읽는 flash 커널과 KV를 다시 쓰는 DynamicCache의 `torch.cat`이 둘 다 공칭 대역폭 근처에서 돈다. 이 cat 복사는 **DFlash와 AR에 공통**이다(DynamicCache의 성질).
- **verify**는 같은 KV를 읽는 데 AR 스텝보다 9배 오래 걸린다(81 GB/s). 원인은 16토큰 query에 bool 마스크가 붙어 mem-efficient 커널로 떨어지는 것과, 그 주변의 복사(110 ms)다. 즉 **8B verify의 긴 S 비용은 대역폭 한계가 아니라 커널 경로 선택의 결과**다.
- **multi-GPU에서의 추가 비용:** verify 마스크는 `cuda:0`에서 만들어지고, 다른 장치에 있는 attention 층마다 복사된다.
  - `cuda:0`이 보내는 P2P 바이트는 8B에서 스텝당 마스크 크기의 24.4–31배다. `cuda:0` 밖에 있는 층 수는 24다. 9B에서는 6.3–11배로, `cuda:0` 밖에 있는 full-attention 층 수(약 6)와 맞는다.
  - 64k에서 8B는 스텝당 25.6 MB, 4.9 ms다[R8, R9 `trace_device.csv` `p2p_memcpy_bytes`].

단계별 변화(4k → 64k):

| 단계 | 8B | 9B | 35B | 병목 |
|---|---|---|---|---|
| prefill(TTFT) | DFlash − AR = +12 → +226 ms | +9 → +314 ms | +10 → +150 ms | 메모리: 피크가 prefill 마지막 층에서 난다(3절) |
| first draft(context 투영) | 15.7 → 187.7 ms | 23.7 → 286.5 | 11.6 → 148.6 | S에 선형. decode의 0.3%, 8.8%, 1.5% |
| steady draft | 5.33 → 13.27 ms | 7.87 → 9.49 | 6.53 → 8.04 | 8B만 S를 따라 커진다(full-attention drafter가 O(S) draft KV를 읽음) |
| draft logits | 2.00 ms 고정 | 3.24 고정 | 1.87 고정 | lm_head 1회. S와 무관 |
| **verify** | 47.8 → 306.5 ms | 59.5 → 100.9 | 183.4 → 257.3 | 8B·9B: attention(GPU). 35B: 커널 launch(host)에 attention이 더해짐 |
| AR 스텝(비교) | 37.7 → 68.5 | 42.0 → 41.9 | 131.7 → 132.2 | |

### 5.2 block 폭 (S=32k)

**근거:** R8, R9(block 조건 행), R35B.

| 모델 | block | DFlash TPOT | acceptance | 확정 토큰/스텝 | verify 1회 | speedup |
|---|---|---|---|---|---|---|
| 8B | 4 / 8 / 16 | 137.3 / 138.2 / 123.0 ms | 0.099 / 0.044 / 0.034 | 1.29 / 1.30 / 1.49 | 166.2 / 168.1 / 171.4 | 0.36 / 0.36 / 0.40× |
| 9B | 4 / 8 / 16 | 23.6 / 24.0 / 23.8 | 0.789 / 0.346 / 0.170 | 3.36 / 3.40 / 3.49 | 65.7 / 67.9 / 69.0 | 1.75 / 1.75 / 1.74× |
| 35B | 4 / 8 / 16 | 65.1 / 38.8 / 35.4 | 0.726 / 0.656 / 0.368 | 3.15 / 5.54 / 6.38 | 195.0 / 204.7 / 214.5 | 2.01 / 3.37 / 3.76× |

- verify 비용은 block 폭에 거의 영향을 받지 않는다(4에서 16으로 가도 +3–10%). verify 1회의 비용은 q가 아니라 KV와 가중치 읽기, 그리고 launch가 정한다.
- 따라서 block 폭의 효과는 **스텝당 확정 토큰 수에서 결정된다.**
  - 35B는 폭을 넓힐수록 확정 토큰이 3.15 → 6.38로 늘어 TPOT이 절반이 된다.
  - 이 문서에서 9B와 8B는 확정 토큰이 폭과 무관하게 거의 같다(9B 약 3.4, 8B 약 1.3). 폭을 넓힌 만큼 거절되는 토큰만 늘어난다.
- R35B의 b16 행(35.37 ms)은 R35의 같은 조건(35.94 ms)과 1.6% 이내로 재현되었다.

---

## 6. 아키텍처별 long-context 병목과 발생 단계

| 아키텍처 | long context에서 커지는 병목 | 발생하는 DFlash 단계 | 성격 | 근거 |
|---|---|---|---|---|
| **Qwen3-8B**(full attention 36) | verify의 attention: bool 마스크 → mem-efficient 커널(119 ms) + 복사(110 ms). verify ÷ AR 스텝 = 4.48(64k) | **verify** | GPU-bound. 대역폭 한계가 아니라 커널 경로 때문(실효 81 GB/s) | R8, T8 |
| | O(S) target KV 9.66 GB, draft KV 1.35 GB(비율 0.139 고정) | 전 단계 상주 | 메모리 | R8 |
| | steady draft가 O(S) draft KV를 읽음(5.3 → 13.3 ms) | steady draft | GPU | R8 |
| | 마스크를 층마다 다른 장치로 복사(스텝당 25.6 MB) | verify(multi-GPU) | 전송 | R8 |
| | acceptance 저하(0.13 → 0.02)와 겹쳐 16k부터 AR보다 느림 | — | — | R8 |
| **Qwen3.5-9B**(GDN 24 + attention 8) | **conv recording buffer 25.8 GB**(attention KV의 12배) + context feature 4.3 GB | **prefill → first draft**(메모리 피크) | 메모리 | R9 |
| | first draft(context 투영) 287 ms = decode의 8.8% | **first draft** | GPU | R9 |
| | verify attention이 S에 비례(6.1 → 70.5 ms). verify ÷ AR 1.41 → 2.41 | **verify** | 짧은 S에서는 host 비중 큼, 긴 S에서는 GPU-bound | R9, T9 |
| | GDN은 S와 무관(8.1 ms/verify) | — | — | T9 |
| **Qwen3.5-35B-A3B**(GDN 30 + attention 10 + MoE) | **커널 launch**: verify 1회 9,239개(expert 루프가 5,808개). 장치 시간 30–52% | **verify**(그리고 AR 자체) | host-bound | R35, T35 |
| | expert GEMM: 층당 48.5 expert hit(AR은 8), 30.6 ms | **verify** | host + GPU | R35 `moe_routing.csv`, T35 |
| | verify attention 6.5 → 83.2 ms(64k에서 최대 항목) | **verify** | GPU | T35 |
| | **conv recording buffer 32.2 GB**(attention KV의 24배) | **prefill → first draft**(메모리 피크) | 메모리. 2 GPU에서는 32k 이상에서 OOM | R35, R35-2 |
| | AR recording 경로의 할당자 단편화(64k에서 장치당 inactive split 13–14 GB) | (기준선 측정) | 할당자 | R35, R35X |

**공통 결론**

1. long context에서 DFlash의 시간 병목은 **verify**다(decode의 81–95%). S와 함께 verify를 키우는 것은 모든 아키텍처에서 **full-attention 층의 verify 경로**(마스크 materialize와 mem-efficient 커널)다. full-attention 층이 적을수록(8B 36층 > 35B 10층 > 9B 8층) 이 증가가 늦게 나타난다.
2. long context에서 DFlash의 메모리 병목은 하이브리드 모델에서 **prefill과 first draft 사이**에 생긴다. rollback을 위해 켠 conv recording과 프롬프트 길이 context feature가 원인이며, draft KV 자체는 작다.
3. MoE 모델은 **S와 무관하게 host-bound**다. S가 커지면 attention 비용이 여기에 더해진다.

---

## 7. 측정상 주의점

- **acceptance는 프로세스 사이에서 재현되지 않을 수 있다.**
  - 같은 64k 조건을 별도 프로세스로 두 번 돌렸을 때, AR native의 greedy 출력이 110번째 토큰에서 갈라졌다. DFlash acceptance는 0.417[R35]과 0.590[R35X]이었다. verify 1회 비용은 257.3 대 256.2 ms로 같았다.
  - 한 프로세스 안의 반복은 결정적이었다. 4k–16k에서는 2 GPU와 3 GPU가 같은 acceptance를 냈다[R35, R35-2].
  - 원인(커널 선택이나 Triton autotune의 프로세스 간 차이)은 확인하지 않았다. S당 프롬프트 1개(K=1), 실행 1회이므로 acceptance와 speedup의 S 추이는 경향으로만 읽는다.
- **64k AR recording의 OOM은 단편화 때문이었다.**
  - 기본 할당자에서는 장치당 reserved 약 50 GB, allocated 33–36 GB, inactive split 13–14 GB로 OOM이 났다[R35].
  - `expandable_segments`를 켜면 allocated 45.0 GB로 성공했고, 오버헤드도 +2.9%(135.9 대 132.1 ms), 편차 ±0.3 ms가 되었다[R35X].
  - 기본 할당자에서 측정한 recording 오버헤드 +9–18%와 큰 편차[R35]는 상당 부분 할당자 효과로 보인다.
- **trace 수치의 성격.**
  - 장치 시간은 profiler 아래에서 decode 앞쪽 32 스텝을 잰 것이다. 커널 실행 시간은 profiler의 영향을 받지 않지만 호스트 시간은 받는다(스텝이 1.85–2.61배 느려짐, `trace_perturbation`).
  - 그래서 host-bound 판정에는 profiler가 없는 perf 벽시계와 trace의 장치 시간을 조합했다.
- **대역폭 수치는 추정이다.** 5.1절의 실효 대역폭은 기록된 커널 시간과 텐서 크기로 계산한 값이며, 하드웨어 카운터로 측정한 값이 아니다.
- verify의 elementwise 복사(8B 64k의 `direct_copy` 110 ms, 35B 64k의 elementwise 37 ms)가 정확히 무엇을 복사하는지는 커널 이름과 소속 모듈(full_attention)까지만 확인했다. 마스크나 입력의 contiguous 변환으로 추정한다.

---

## 8. 재현

```bash
# 측정 (3 GPU)
python -m dflash.arch --num-gpus 3 sweep qwen3-8b        --sweeps sequence block --no-natural --passes perf memory trace --repeats 3
python -m dflash.arch --num-gpus 3 sweep qwen3.5-9b      --sweeps sequence block --no-natural --passes perf memory trace --repeats 3
python -m dflash.arch --num-gpus 3 sweep qwen3.5-35b-a3b --sweeps sequence --no-natural --passes perf memory trace moe --repeats 3
python -m dflash.arch --num-gpus 3 sweep qwen3.5-35b-a3b --sweeps sequence block --input-tokens 32768 --no-natural --passes perf memory trace moe --repeats 3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m dflash.arch --num-gpus 3 sweep qwen3.5-35b-a3b --sweeps sequence --input-tokens 65536 --no-natural --passes perf memory --repeats 3

# 표 재생성과 모듈 × 커널 분해
python -m dflash.arch export <record.json>
python -m dflash.arch trace-breakdown <trace.json.gz> --phase "decode: target verify"
python -m dflash.arch trace-breakdown <trace.json.gz> --phase "decode: ar step"
```

실행 스크립트는 `queue/run_arch_35b_3gpu.sh`, `queue/run_arch_35b_3gpu_64k_expandable.sh`, `queue/run_arch_v2_research_queue.sh`이고, 로그는 `queue/arch_*_3gpu*.log`, `queue/arch_v2_*.log`다.
