# 측정 코드 감사: 중복 측정, 측정 누락, GPU 활용률 미측정

- 대상: `main`의 `dflash/arch/*`와 `record_arch_main/`의 스윕 기록 7개. 8B, 9B, 35B는 FLA arm, 9B와 35B는 no-FLA arm도 포함합니다.
- 작성일: 2026-09-30
- 성격: 검사 보고서입니다. 코드는 수정하지 않았습니다. 수정안은 각 항목에 적었습니다.

> **개정 이력.** 초판은 `dflash/model.py`를 읽지 못한 상태에서 작성되었다. 이후 사용자 허용으로 읽고 개정했다: A4를 정정했고(중복 측정 아님, 감사자 계산 오류), A8을 추가했으며, A2·A5·A7의 `model.py` 쪽 근거를 보강했다.

---

## 0. 요약

| # | 분류 | 결함 | 틀리게 만드는 결과 | 심각도 |
|---|---|---|---|---|
| A1 | 측정 오류 | AR 기준선을 drafter가 GPU에 올라간 같은 프로세스에서 측정 | AR 메모리 기준선, drafter 메모리 오버헤드, 35B 64k AR OOM | **높음** |
| A2 | 측정 오류 | AR과 DFlash의 `peak_memory_per_device_bytes` 정의가 다름 | 장치별 메모리 비교 전부 | **높음** |
| A3 | 교란 변수 | S마다 다른 LongBench 과제를 쓰고, S마다 프롬프트가 1개뿐 | S축 acceptance, speedup 곡선 | **높음** |
| A4 | 누락/단위 혼동 | (정정: 중복 없음) draft logits 등 스텝당 2.6–4 ms가 미귀속, 타이머 필드 단위 혼재 | draft 비중(하한으로만 유효) | 중간 |
| A5 | 정의 불일치 | TPOT 분모가 DFlash는 n, AR은 n−1 | speedup(+0.39% DFlash 쪽으로 편향) | 낮음 |
| A6 | 기록 손상 | no-FLA 실행이 모델별 `csv/`를 덮어씀 | 9B, 35B CSV 전부(JSON은 온전함) | **높음**(재생성으로 복구 가능) |
| A8 | 비대칭 계측 | PeakTracker 층별 훅(40.9 µs×층수)이 perf 패스 DFlash에만 걸림 | speedup을 DFlash에 불리하게 1–3% 과소 | 중간 |
| A7 | 순서 의존 | AR 정책 사이에 allocator를 정리하지 않음. peak는 allocated만 기록 | 35B 64k recording OOM 판정 | 중간 |
| B1 | 미측정 | **GPU 활용률을 전혀 측정하지 않음**(NVML 없음, 커널 타임라인 없음, trace/counters 패스는 CLI에 없음) | "어느 GPU가 언제 노는가"와 bandwidth/compute 판정 | **높음** |
| B2 | 미측정 | verify 내부를 층/mixer/연산으로 쪼개지 않음 | 가장 큰 DFlash 병목(긴 S에서 verify 비용 증가)의 원인 | **높음** |
| B3 | 미측정 | MoE 라우팅과 expert 비용 캡처 코드가 어디에도 연결되지 않음 | 35B의 AR 135 ms/token과 verify 비용의 원인 | **높음** |
| B4 | 미실행 도구의 결함 | `verify-width` 벤치가 무작위 토큰, filler 프롬프트, recording 정책을 씀 | 실행하면 MoE 결과가 비대표적이 됨 | 중간 |
| B5 | 미측정 | 3-GPU 샤딩의 장치 간 전송과 대기 비용 | 절대 TPOT, verify와 draft의 비율 | 중간 |
| C | 죽은 코드 | `events.py`, `nvtxr.py`, `PASS_TRACE/COUNTERS`, `routing_rows`, 틀린 docstring 2곳 | 존재한다는 이유로 측정되고 있다고 오해하기 쉬움 | 낮음 |

---

## A. 결과 수치를 틀리게 만드는 결함

### A1. AR 기준선에 drafter가 GPU 상주

**위치**
- `dflash/arch/cli.py:188`: `loader.load_pair(..., with_draft=True)`로 target과 drafter를 함께 올립니다.
- `cli.py:232-242`: 같은 프로세스에서 `run.run_ar`를 실행합니다.
- 코드와 모순되는 docstring:
  - `loader.py:10-13`: "The AR pass runs in a process that never calls load_draft"
  - `run.py:254-258`: "This is the whole reason AR runs in its own process"
  - `ar.py:5-9`

**어떻게 틀리는가.** `run_ar`는 `torch.cuda.max_memory_allocated(i)`를 읽습니다(`run.py:266-273`). 이 값은 프로세스에 상주하는 모든 텐서를 포함하므로 drafter 가중치도 들어갑니다. drafter 가중치는 8B 2.10 GB, 9B 2.58 GB, 35B 0.77 GB이고 모두 cuda:0에 있습니다. drafter가 추가로 쓰는 메모리를 "DFlash − AR"로 구하면 이 가중치만큼 과소평가됩니다.

**데이터 근거.** 8B S=4k에서 cuda:0의 AR native 피크는 8.62 GB로, 같은 장치의 DFlash 값 8.39 GB보다 큽니다. drafter가 없는 기준선이라면 나올 수 없는 값입니다. 여기에는 A2도 함께 작용합니다.

**영향**
- `speedup.csv`의 `ar_*_peak_sum_device_maxima_bytes`
- 연구 질문 1(drafter 메모리 비중)의 분모
- 35B 64k AR `recording`의 OOM도 drafter가 상주한 상태에서 난 것입니다.

**수정안(둘 중 하나)**
1. AR을 별도 프로세스에서 `with_draft=False`로 실행합니다.
2. 같은 프로세스를 유지하되, AR 실행 전에 drafter를 CPU로 내리고 `empty_cache`를 호출한 뒤 `drafter_resident: false`를 기록합니다.

### A2. 장치별 피크의 정의가 두 경로에서 다름

**위치**
- AR: `run.py:305-306`. `peak_memory_per_device_bytes`는 **장치마다 run 전체 동안의 최댓값**입니다.
- DFlash: `run.py:126`, 값은 `model.py:433`의 PeakTracker(`peak_per_device`)에서 옵니다. 같은 이름의 필드가 **합계가 최대인 순간의 장치별 값**입니다. `report.device_rows`의 컬럼명 `peak_at_aggregate_peak_bytes`가 이를 확인해 줍니다.

**어떻게 틀리는가.** 이름이 같은 두 필드를 나란히 놓으면 정의가 다른 값을 빼거나 비교하게 됩니다. 35B의 DFlash 합계 기준 두 값은 116.8 GB(동시 피크)와 134.6 GB(장치별 최댓값의 합)로 18 GB 차이가 납니다.

**정정.** 직전 35B 결과 보고에서 "AR native 피크 32.6/29.8/32.5 GiB, DFlash 35.5/30.8/42.5 GiB"라고 나란히 비교했습니다. 이 비교는 정의가 다른 값끼리의 비교이므로 무효입니다. 비교할 수 있는 값은 `peak_memory_sum_device_maxima_bytes`끼리이고, 이 값도 A1 때문에 AR 쪽에 drafter가 포함되어 있습니다.

**수정안.** DFlash 경로에도 장치별 최댓값을 `peak_memory_device_maxima_bytes`라는 별도 필드로 기록합니다. 현재 필드는 `peak_memory_per_device_at_aggregate_peak_bytes`로 이름을 바꿉니다.

### A3. S 스윕이 프롬프트 과제와 교란됨

**위치.** `cli.py:215-218`에서 `prompts.build(tokenizer, S)`를 호출합니다. 이 함수는 S마다 `context.build_dataset(..., 1, seed=42)`로 샘플 **1개**를 새로 뽑습니다(`prompts.py:54-64`).

**실제 배정(세 모델 동일)**

| S | 과제 / source_index | 비고 |
|---|---|---|
| 4k | multi_news / 410 | |
| 8k | 2wikimqa / 2073 | |
| 16k | **trec** / 2990 | few-shot 분류 과제 |
| 32k | 2wikimqa / 2015 | composed |
| 64k | 2wikimqa / 2015 | composed |

**어떻게 틀리는가.**
- acceptance는 내용에 의존합니다. 따라서 S축의 acceptance 변화와 speedup 변화에는 **S의 효과와 과제의 효과가 섞여 있습니다.** 35B 16k에서 acceptance가 0.091로 떨어지고 speedup이 1.53×인 것이 대표적입니다. 9B 16k에서 speedup이 4.72×로 가장 높은 것도 같은 과제에서 나왔습니다.
- `repeats=3`은 greedy 결정적 반복이라 표준편차(0.01–1%)는 **타이밍 노이즈만** 나타냅니다. 프롬프트 간 분산은 한 번도 측정되지 않았습니다(n=1).

**수정안**
- S축에서는 같은 문서를 앞에서부터 잘라 쓰는 **중첩(prefix) 프롬프트**를 사용하고, 과제는 하나로 고정합니다.
- 또는 S마다 K≥5개의 프롬프트를 쓰고 프롬프트 간 분산을 따로 보고합니다.
- latency와 memory처럼 내용에 무관한 지표도 **step 수가 acceptance로 정해지므로** 교란됩니다. verify 한 번당 비용처럼 step으로 정규화한 지표를 함께 봐야 합니다.

### A4. (정정) 단계 시간 중복은 없음 — 대신 스텝당 2.6–4 ms가 어느 단계에도 귀속되지 않고, draft 비용이 과소 집계됨

초판의 "단계 합이 벽시계를 최대 7.5% 초과(중복 측정)"는 **감사자의 계산 오류**였다. `context_feature_s`는 `dflash_generate`가 누적 합계로 반환하는데(`model.py:1227`) 이를 스텝당 평균으로 보고 스텝 수를 곱했다. `model.py`를 읽은 뒤 다시 계산하면:

```
unattributed = decode_latency_s − (target_forward_s + context_feature_s
                                   + first_draft_forward_s + mean_steady_draft_forward_s·(steps−1))
```

| 모델 | S=4k | 8k | 16k | 32k | 64k | 스텝당 |
|---|---|---|---|---|---|---|
| 8B | +5.1% | +4.0% | +2.5% | +1.5% | +0.9% | 2.8 ms |
| 9B | +5.4% | +5.3% | +5.1% | +4.5% | +2.7% | 3.6–4.0 ms |
| 35B | +1.7% | +1.5% | +1.4% | +1.1% | +1.0% | 2.6–3.3 ms |

잔차는 전부 양수이고 스텝당 거의 상수다. 중복은 없다. 남는 문제:

- **draft 비용이 과소 집계된다.** `draft_forward_timer`는 drafter 본체 호출(`model.py:1008-1021`)만 감싼다. 그 뒤의 `compute_logits`(`model.py:1051-1053`)는 **target의 lm_head**를 쓰고, 샤딩된 target에서 lm_head는 마지막 GPU에 있으므로 cuda:0 → cuda:2 → cuda:0 왕복과 15×vocab GEMM이 따라온다. 이 구간은 어떤 타이머에도 없다. noise embedding, crop, `.item()` 동기화, accelerate 훅, A8의 PeakTracker 훅도 마찬가지다. 결과적으로 표의 "draft 비중"(9B 11–15%)은 하한이다.
- **단위가 섞여 있다.** 같은 행에서 `target_forward_s`, `context_feature_s`, `first_draft_forward_s`는 합계, `mean_steady_draft_forward_s`는 호출당 평균이다. 게다가 `context_feature_timer`는 prefill의 feature 구축(`model.py:923, 940`)까지 합산하므로 decode 구간 지표와 섞인다. 이번 감사의 오류가 바로 이 혼동에서 나왔다.
- 잔차를 기록하는 필드가 없다. `events.residual()`(`events.py:216`)은 미사용.

**수정안:** draft logits(`compute_logits`/`propose`)를 별도 타이머로 감싸 `draft_logits_s`로 기록; 모든 타이머 필드를 `*_total_s`/`*_mean_s`로 이름 통일; prefill과 decode의 context-feature 타이머 분리; `unattributed_decode_s`를 레코드에 추가.

**타이머 자체의 타당성(확인됨).** `_GpuTimer`는 현재 장치(cuda:0) 스트림에 CUDA event를 기록한다(`model.py:638-651`). target이 세 GPU에 걸쳐 있어도 accelerate의 최상위 훅이 출력(logits)을 입력 장치로 되돌리는 복사를 하므로 cuda:0 스트림이 마지막 shard의 완료를 기다리고, 따라서 `target_forward_s`는 verify 전체를 포함한다. 잔차가 양수이고 작다는 데이터가 이를 뒷받침한다.

### A5. TPOT 분모 불일치

- DFlash: `model.py:1200`의 `total_decode_time / num_output_tokens`. n=256입니다.
- AR: `ar.py:262-264`에서 `decode_time / (executed − 1)`이고 n−1=255입니다.
- 두 경로 모두 첫 토큰은 prefill(TTFT)에서 나오므로 DFlash 쪽 분모가 1 큽니다.
- 결과적으로 speedup이 256/255, 즉 0.39% 과대평가됩니다. 크기는 작지만 체계적입니다.
- **수정안:** 두 경로 모두 `decode / (n−1)`을 쓰도록 통일합니다.

### A6. 모델별 `csv/` 디렉터리가 덮어써짐

**위치.** `cli.py:299-301`의 `report.export(payload, <root>/<model>/csv)`. 디렉터리가 레코드 라벨(backend)과 무관하게 모델마다 하나입니다.

**현재 상태**
- `record_arch_main/qwen3.5-9b/csv/*.csv`와 `qwen3.5-35b-a3b/csv/*.csv`에는 no-FLA arm의 **S=32k 한 행만** 남아 있습니다(`speedup.csv` 2줄). 기본 arm인 FLA의 5개 S는 CSV에서 사라졌습니다.
- 8B CSV는 온전합니다.
- JSON 레코드는 모두 온전하므로 `report.export`로 다시 만들 수 있습니다.

**수정안**
- 출력 경로를 `csv/<record 파일명>/`처럼 레코드별로 둡니다.
- FLA 레코드에서 CSV를 재생성합니다(재생성하는 데 코드 변경은 필요 없음).

### A7. AR recording OOM이 실행 순서에 의존

**위치.** `cli.py:232-242`에서 native(웜업 1 + 반복 3) 직후 recording을 실행합니다. 그 사이에 `empty_cache`나 allocator 통계 초기화가 없습니다. `empty_cache`는 조건 단위로만 호출됩니다(`cli.py:275`).

**어떻게 틀리는가**
- 35B 64k recording이 OOM 났을 때 GPU 2에는 **13.03 GiB가 reserved-but-unallocated** 상태였습니다. 앞선 실행이 남긴 단편화와 상주 drafter(A1)가 원인일 가능성이 있습니다.
- `dflash_generate`는 `peak_memory_reserved_bytes`를 계산하지만(`model.py:1220`) `run._stats_to_row`가 이를 행에 옮기지 않고, AR 경로는 아예 계산하지 않습니다. 기록에 남는 피크가 allocated뿐이라 OOM을 결정하는 reserved 값과 단편화는 남지 않습니다. 따라서 "용량 한계"인지 "allocator 상태"인지 기록만으로는 판정할 수 없습니다.

**수정안**
- 정책마다 `empty_cache`와 `reset_peak_memory_stats`를 호출합니다.
- `max_memory_reserved`와 OOM 시점의 `memory_stats()`를 기록합니다.
- 두 정책의 실행 순서를 교대로 바꾸거나 별도 프로세스에서 실행합니다.

### A8. PeakTracker의 층별 훅이 perf 패스의 DFlash에만 걸림

- **위치:** `model.py:883-885, 955-960`. `return_stats=True`이면 perf 패스에서도 PeakTracker가 켜지고, target이 여러 장치에 있으면 **모든 decoder layer에 pre-hook**이 남는다(단일 장치일 때만 제거). 각 훅은 장치마다 allocator 통계를 읽고 peak를 리셋한다(`model.py:414-445`).
- **측정한 비용:** 사용 가능한 GPU 3장에서 `boundary()` 1회 = **40.9 µs**(host). target forward 1회당 8B 1.47 ms, 9B 1.31 ms, 35B 1.64 ms. 여기에 스텝당 phase 경계 약 7회(약 0.3 ms)가 더해진다.
- **어떻게 틀리는가:** AR 경로(`ar.generate`)에는 이 훅이 없다. B=1의 작은 커널에서 host가 병목이면 이 시간이 그대로 DFlash 스텝 시간에 더해진다. 상한으로 잡으면 DFlash TPOT이 8B 4k 약 3%, 9B 약 2%, 35B 약 1% 부풀고, 그만큼 speedup이 **DFlash에 불리하게** 과소평가된다. perf와 memory 두 패스 모두에 훅이 있으므로 `pass_comparison`의 섭동 검사로는 드러나지 않는다.
- **수정안:** perf 패스에서는 `PeakTracker(enabled=False)` 또는 층 훅 없이 phase 경계만 사용하고, 층 단위 피크는 memory 패스에서만 잰다. 또는 AR에도 같은 훅을 걸어 대칭으로 만든다.

---

## B. 측정되지 않아 병목을 비교할 수 없는 지점

### B1. GPU 활용률: 측정 경로가 없음

**현황**
- `dflash/arch`에는 NVML 샘플링, `torch.profiler`, CUPTI, Nsight를 연동하는 코드가 **하나도 없습니다.** 이 패키지에서 `nvidia-smi`는 manifest에만 쓰입니다.
- `events.py`에는 gap 분류(`structural_shard_wait`, `host_launch_or_sync`, `transfer` 등)와 bottleneck 라벨(`bandwidth`, `compute`, `host_launch` 등)이 정의되어 있습니다. 하지만 `EventLog`는 어디서도 생성되지 않으므로 이 라벨이 붙은 기록은 **0건**입니다.
- `run.py:47-48`에 `PASS_TRACE`와 `PASS_COUNTERS`가 정의되어 있지만 `--passes` 선택지(`cli.py:374-375`)에 없어 실행할 수 없습니다.
- `nvtxr.py`는 `dflash/arch` 안의 어떤 모듈에서도 import되지 않습니다(`model.py`는 확인하지 못함).

**그래서 무엇을 모르는가**
- B=1에서 3-GPU로 layer를 샤딩하면 구조상 한 순간에 대략 한 장치만 일합니다. 장치별 busy 비율, 장치 사이 대기, host launch 공백을 구분할 방법이 없습니다.
- `nvidia-smi`의 `utilization.gpu`는 "샘플 구간에 커널이 하나라도 돈 시간의 비율"입니다. SM 점유율도 HBM 대역폭 사용률도 아니므로 이것으로 병목을 판정할 수는 없습니다.

**기록에서 역산한 추정치(측정이 아님)**

| 경로 | 계산 | 단일 A6000 공칭 대비 |
|---|---|---|
| 8B AR 4k | 가중치 약 16 GB ÷ 36.5 ms = 약 450 GB/s | 768 GB/s의 약 58% |
| 35B AR | active 약 3B, 약 6 GB 미만 ÷ 135 ms = 약 45 GB/s | **약 6%** |

35B AR은 대역폭 한계도 연산 한계도 아닐 가능성이 큽니다. host launch 또는 Python expert 루프를 의심할 수 있지만 측정된 근거는 없습니다.

**수정안(패스 1개 추가)**
- `torch.profiler`(CUDA activity) 또는 `nsys`로 **장치별 커널 구간의 union**을 구하고 decode 벽시계 시간으로 나눈 값을 `device_busy_fraction`으로 기록합니다. `events.union_seconds`가 이 계산용으로 이미 있습니다.
- 여기에 커널 수(launch 수)와 phase별 DRAM 바이트를 추가합니다(`ncu`, 대표 조건 1개만). 그러면 phase마다 achieved bandwidth와 공칭 대역폭을 비교할 수 있습니다.

### B2. verify 내부가 한 덩어리로만 기록됨

**현황.** `target_forward_s`는 verify 한 번 전체의 시간입니다. 층, mixer(full-attention/GDN), FFN(dense/MoE), lm_head로 쪼개지지 않습니다.

**데이터가 보여 주는 가장 큰 병목.** verify 한 번(q=16)의 비용과 AR 한 스텝(q=1)의 비용을 비교했습니다.

| 모델 | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| 8B verify / AR | 47 / 36 ms (1.30×) | 62 / 36 (1.72×) | 103 / 39 (2.62×) | 172 / 50 (3.45×) | **306 / 69 (4.47×)** |
| 9B | 62 / 41 (1.52×) | 62 / 41 (1.52×) | 62 / 41 (1.52×) | 68 / 41 (1.65×) | 101 / 41 (2.44×) |
| 35B | 187 / 136 (1.38×) | 192 / 134 (1.43×) | 195 / 135 (1.45×) | 223 / 137 (1.63×) | 261 / 135 (1.93×) |

- 8B에서 S에 비례하는 증가분은 verify가 1k 토큰당 약 3.6 ms, AR이 약 0.55 ms로 **약 6.6배**입니다.
- 16 query가 같은 KV를 한 번 읽는 memory-bound attention이라면 q=1과 비슷해야 합니다. 따라서 verify의 attention이 flash/memory-efficient 커널 대신 마스크를 materialize하는 경로를 탈 가능성이 있습니다.
- 8B가 64k에서 0.30×(AR보다 느림)로 떨어지는 주원인으로 보이지만, 현재 코드로는 **층이나 연산 단위로 입증할 수 없습니다.**
- 9B에서는 full-attention이 8개 층뿐이라 증가가 64k에서야 보입니다.

**수정안**
- 대표 조건에서 layer, mixer, FFN 단위 CUDA event(동기화 없음)를 넣은 **분해 패스**를 추가합니다.
- attention 호출의 SDPA backend(flash/efficient/math)와 마스크 shape를 기록합니다.
- `verify-width`(B4를 고친 뒤)를 S∈{4k, 32k, 64k}에서 실행합니다.

### B3. MoE 비용: 캡처 코드가 연결되지 않음

**현황.** `moe.RoutingCapture`(`moe.py:47`)는 어떤 CLI 서브커맨드에서도 import되지 않습니다. `report.routing_rows`도 `EXPORTERS`에 등록되어 있지 않습니다(`report.py:329-336`). 연구 질문 2의 route, dispatch, expert GEMM 분해는 **한 번도 실행된 적이 없습니다.**

**데이터로 좁혀지는 것**
- 35B AR은 FLA arm에서 137.0 ms/token, no-FLA arm에서 137.9 ms/token입니다(S=32k). GDN 커널을 바꿔도 1% 미만이 변하므로 **35B AR 비용의 대부분은 GDN이 아닙니다.**
- 35B의 verify/AR 비율은 짧은 S에서도 1.38×입니다. q=16이면 hit expert 수가 늘어 expert 루프가 길어진다는 가설과 맞지만, 측정된 것은 아닙니다.

**수정안.** `sweep`에 `--passes moe`(RoutingCapture를 켠 1회 실행)를 추가하고 `routing` CSV exporter를 등록합니다. 동기화 오버헤드가 있으므로 perf 패스와는 분리합니다(모듈 docstring대로).

### B4. `verify-width` 벤치의 결함(아직 실행되지 않음)

- `verifyq.py:146-149`: 검증 블록이 **무작위 token id**입니다. dense 모델은 비용이 내용과 무관하지만, MoE는 라우팅이 내용을 따르므로 hit expert 분포가 실제 draft 블록과 다릅니다.
- `cli.py:120`: 프롬프트가 LongBench가 아니라 반복 filler 텍스트입니다. `prompts.build`와 다른 경로입니다.
- `verifyq.py:103`: 기본값이 `cache_policy_recording=True`이므로 q=1 행은 AR native 경로가 아니라 `causal_conv1d_fn` 일반 경로를 탑니다(`ar.py` docstring 참고). "q=1 = AR 스텝"이라고 읽으면 틀립니다.
- **수정안:** 실제 DFlash run에서 캡처한 draft 블록을 재사용합니다. `prompts.build`를 쓰고, q=1 행은 native 정책으로도 측정합니다.

### B5. 샤딩의 전송과 대기 비용

- 모든 모델을 3-GPU로 샤딩했고 drafter는 cuda:0에 있습니다. 매 토큰(AR)과 매 verify마다 장치 경계 2곳에서 hidden 전송과 동기화가 일어나지만 이 비용은 따로 측정되지 않습니다.
- 참고 수치(코드 버전이 달라 **지표용**일 뿐): 9B 단일 장치 레코드(`sweep_20260929-081633`)에서 AR은 31.7 ms/token, DFlash는 7.86 ms/token이었습니다. 이번 3-GPU 레코드에서는 각각 40.6과 10.1 ms/token입니다. 비율은 거의 같지만(4.04× 대 4.00×) 절대 시간은 약 25% 부풀었을 수 있습니다.
- **수정안:** B1의 트레이스에서 장치 경계의 P2P copy 구간과 대기 공백을 분리합니다. 8B와 9B는 짧은 S에서 단일 장치 대조점을 1개 둡니다.

---

## C. 죽은 코드와 틀린 문서(측정값에는 영향 없음)

- `events.py`: `EventLog`, `union_seconds`, `residual`, gap/bottleneck 상수가 정의만 되어 있고 호출되지 않습니다. B1과 A4의 수정안에서 재사용할 수 있습니다.
- `nvtxr.py`: `dflash/arch` 안에서 import되지 않습니다.
- `run.py:47-48`: `PASS_TRACE`, `PASS_COUNTERS`. docstring(`run.py:22-24`)은 실행 가능한 것처럼 설명합니다.
- `report.routing_rows`: 등록되지 않았습니다.
- `loader.py:10-13`, `run.py:254-258`, `ar.py:5-9`: "AR은 drafter가 없는 프로세스에서 실행된다"는 설명이 사실과 다릅니다(A1).

---

## D. 이번 검사에서 문제가 없음을 확인한 항목

- **probe 섭동:** memory 패스의 TPOT 섭동은 −0.02%~+2.9%이고, 모든 조건에서 `acceptance_identical=True`입니다.
- **lossless 감사:** `audit_follows_perf=True`(전 조건), 전부 수락된 step의 replay 오차는 0.0입니다. audit와 exact 패스의 타이밍은 결과표에 쓰이지 않습니다.
- **AR 루프:** decode 루프 안에 host sync가 없습니다(`ignore_eos` 조건). TTFT와 decode 구간은 `synchronize`로 닫힙니다.
- **statemem:** 같은 storage를 한 번만 계산하고(alias를 기록), conv working과 recording을 분리해 계산합니다.
- **backend 라벨:** FLA arm은 `causal_conv1d+fla`, no-FLA arm은 `torch_reference`로 레코드 파일명과 JSON에 모두 들어 있습니다.

---

## E. 수정 우선순위(제안)

1. **A6** CSV 재생성과 경로 분리. 가장 싸고, 이미 손상된 산출물을 복구합니다.
2. **A1, A2, A7** AR 메모리 기준선 재설계. drafter를 빼고, 정의를 통일하고, reserved도 기록합니다. 그다음 35B 64k recording을 재측정합니다.
3. **A3** S축을 prefix 프롬프트로 재구성하거나 K개 프롬프트로 늘립니다. 그 전까지 S축 speedup은 "S와 과제의 혼합 효과"로 표기합니다.
4. **B2, B1** verify 분해 패스와 장치 busy 트레이스 패스. 8B와 9B의 긴 S verify 병목, 샤딩 대기를 확인합니다.
5. **B3, B4** MoE 캡처를 연결하고 verify-width를 고쳐 실행합니다.
6. **A8** perf 패스에서 층별 PeakTracker 훅을 끕니다(한 줄 수정, speedup 1–3% 보정).
7. **A4, A5** draft logits 타이머를 추가하고, 타이머 필드 단위를 통일하고, TPOT 분모를 맞춥니다.

---

## F. 수정 상태 (2026-09-30, measurement protocol v2)

코드는 `dflash/arch/*`, `dflash/model.py`(계측 스위치만 추가, 기본값은 기존 동작)에서 수정했다. 새 기록에는 `measurement_protocol` 블록과 `gpu_selection`이 들어간다. 모든 GPU 서브커맨드는 전역 `--num-gpus N`(기본 2)으로 N장을 UUID로 고정한다.

| # | 상태 | 수정 내용 |
|---|---|---|
| A1 | 수정 | `sweep`이 모든 AR 조건을 target만 올린 상태에서 먼저 돌리고, 그다음에 drafter를 로드함. AR 행에 `drafter_resident: false` 기록 |
| A2 | 수정 | `peak_memory_device_maxima_bytes`(두 경로 동일 정의)와 `peak_memory_per_device_at_aggregate_peak_bytes`(memory 패스만)로 분리 |
| A3 | 부분 수정 | `--prompt-mode fixed-task`(기본): 과제, split, 확장 정책을 고정하고 모든 S에서 같은 source 문서를 씀(`same_source_across_lengths`로 검증). `--prompts-per-length K`로 프롬프트 간 분산을 측정. 스텝 정규화 지표 `verify_over_ar_step` 추가. **내용과 S의 완전 분리는 불가능**(아래 참고) |
| A4 | 수정 | `draft_logits_total_s` 타이머, prefill/decode context-feature 분리, `*_total_s`/`*_mean_s` 명명, `unattributed_decode_s` 기록 |
| A5 | 수정 | 두 경로 모두 TPOT = decode/(n−1) |
| A6 | 수정 | CSV를 `csv/<record 이름>/`에 씀. `export` 서브커맨드 추가. 기존 5개 레코드의 CSV를 재생성함 |
| A7 | 수정 | 패스마다 `empty_cache`와 peak 리셋, AR 정책 순서 교대(`--ar-order alternate`), reserved 피크와 OOM 시점의 allocator 스냅샷(`inactive_split_bytes`) 기록. 추가로 발견한 것: PeakTracker가 켜진 실행에서는 allocator peak 카운터가 경계마다 리셋되므로, 기존 `peak_memory_reserved_bytes`는 마지막 구간의 값일 뿐이었음 |
| A8 | 수정 | perf 패스는 `memory_tracking=False, watch_layers=False`로 실행. 메모리 값은 memory 패스에서 가져옴 |
| B1 | 수정(CUPTI) | `--passes trace`: torch.profiler로 장치별 busy, 다른 장치가 일하는 동안의 대기(shard wait), 모든 장치 유휴, P2P memcpy를 측정. AR과 DFlash 모두 기록 |
| B2 | 수정 | trace 패스의 module range로 phase×(mixer/FFN/MoE 하위/lm_head)×layer 장치 시간을 분해. 커널 이름과 SDPA 마스크 기록 |
| B3 | 수정 | `--passes moe`: AR 스텝과 verify의 expert 라우팅. `moe_routing.csv` |
| B4 | 수정 | verify-width가 fixed-task 프롬프트, target greedy continuation 블록, native와 recording 두 정책을 사용 |
| B5 | 수정 | trace 패스의 `p2p_memcpy_s/bytes`, `idle_shard_wait_s` |
| C | 정리 | 틀린 docstring 수정, `events.union_seconds`를 trace에서 사용 |

8B, 2 GPU, S=4k 스모크 결과(측정 1회, 참고용): verify는 SDPA에 bool 마스크 `[1,1,16,S]`를 넘겨 mem-efficient 커널(8.4 ms)과 elementwise 11.5 ms를 쓴다. AR은 마스크 없이 flash 커널(1.4 ms)을 쓴다. FFN은 16.5 대 16.1 ms로 같다. B2의 가설이 커널 수준에서 확인됐다.

### F.1 v2 적용 중 새로 발견한 측정 오류 (수정함)

- **profiler 잔류 오버헤드.** torch.profiler를 한 번 실행한 프로세스는 이후의 모든 kernel launch가 느려진다. Qwen3-8B 단일 GPU AR TPOT이 29.4 ms에서 33.2 ms로(+13%), 35B-A3B 2 GPU에서는 132 ms에서 164 ms로(+24%) 늘었고, 프로세스가 끝날 때까지 유지됐다. 처음 구현한 v2는 조건마다 AR trace를 AR 시간 측정 직후에 돌렸다. 그래서 두 번째 조건부터의 모든 시간 측정이 오염되었다. 첫 8B 스모크 기록에서도 DFlash perf가 AR trace 뒤에 돌아서 speedup이 과소평가됐다. 이제는 모든 시간 측정 패스가 먼저 돌고 진단 패스는 마지막에 돌며, 행마다 `profiler_used_before`가 기록된다.
- **전체 실행 트레이스는 확장되지 않는다.** 35B의 S=4k AR 전체 트레이스가 gzip으로 1 GB였고, 파싱에 호스트 메모리 100 GB 이상을 썼다. 이제 trace는 decode 앞쪽의 `--trace-steps`(기본 32) 스텝만 기록한다.
- **memcpy가 busy에서 빠져 있었다.** memcpy 이벤트는 `device`가 아니라 `inDevice` 필드에 장치를 적는다. 이를 반영해 수정했다.
