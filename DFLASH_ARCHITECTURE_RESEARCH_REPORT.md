# DFlash를 GDN·MoE에 적용할 때의 병목 이동과 연구 질문

작성일: 2026-10-01. 분석 대상은 **B=1**이며 배치 크기 실험은 제외한다. 기존 GPU 실험의 JSON·CSV, 커널 분해 결과와 측정 코드를 재분석한 보고서다. 새로운 GPU 성능 실험이나 rollback 수정은 수행하지 않았다.

## 1. 연구 결론과 판단 범위

**현재 자료가 지지하는 주장은 “KV 병목이 compute 병목으로 일괄 이동한다”가 아니다. Target의 KV가 작아지면서, speculation을 위해 과거를 보존하고 수락 경계의 상태를 확정하는 비용이 드러나며, MoE에서는 expert 집합의 확장과 작은 연산의 실행 비용이 추가된다. 남아 있는 full-attention 층은 긴 context에서 다시 시간 병목이 된다.**

따라서 연구 목표는 다음과 같이 구체화하는 것이 적절하다.

> **작은 recurrent state와 sparse expert 실행이라는 target의 효율성을, 정확한 speculative verification과 state commit을 지원하면서도 유지할 수 있는가?**

| 검증하려던 가설 | 현재 데이터에 근거한 판정 | 논문에서 주장할 수 있는 범위 |
|---|---|---|
| ① GDN으로 target KV가 감소하면 DFlash overhead의 상대 비중이 증가한다 | **지지됨. 단, 가장 큰 항목은 steady drafter KV가 아니라 prefill conv recording과 prompt conditioning feature** | 현재 구현에서 생기는 일시적 메모리 증폭. 모든 GDN 구현의 필연적 비용이라고 일반화하면 안 됨 |
| ② MoE의 parallel verification이 expert activation 비용을 증가시킨다 | **지지됨.** Expert union, expert GEMM 시간, launch 수가 증가 | 현재 HF expert 실행 경로에서 확인. 최적화된 fused/grouped MoE의 성능은 미측정 |
| ③ memory-bound에서 compute-bound로 전환한다 | **미입증.** GPU 실행 비중 증가와 compute 포화는 다름 | 하드웨어 카운터가 없고, 짧은 block에서 expert당 토큰 수도 작음. host/launch·weight traffic·잔존 attention을 분리해야 함 |
| GDN rollback을 사용하면 target 결과가 보존된다 | **측정 구현에서는 성립하지 않음.** Recurrent state 복구 누락이 확인됨 | 현재 GDN speedup은 stock 구현의 실행 비용이며, lossless DFlash의 확정된 speedup이 아님 |

가장 먼저 해결할 문제는 rollback 정확성이다. 이것은 성능과 별개의 부록이 아니라, acceptance·expert routing·실제 확정 토큰 수의 해석을 바꾸는 전제 조건이다.

## 2. 자료 출처와 비교 조건

아래 ID는 원본 [LONG_CONTEXT_BOTTLENECK_REPORT.md](LONG_CONTEXT_BOTTLENECK_REPORT.md)의 명칭을 유지한다. 경로를 클릭하면 해당 원본을 확인할 수 있다.

| ID | 원본 | 본 보고서에서의 용도 |
|---|---|---|
| R8 | [8B v2, 20260930-113850](record_arch_main/qwen3-8b/sweep_causal_conv1d+fla_3gpu_20260930-113850.json) | Dense attention 비교군, S·block sweep |
| R9 | [9B v2, 20260930-103640](record_arch_main/qwen3.5-9b/sweep_causal_conv1d+fla_3gpu_20260930-103640.json) | GDN+dense, S·block sweep |
| R35 | [35B v2, 20260930-023904](record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_3gpu_20260930-023904.json) | GDN+MoE, S sweep와 routing |
| R35B | [35B block sweep, 20260930-091225](record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_3gpu_20260930-091225.json) | 32K, block 4·8·16, expert union 재집계 |
| R35X | [35B allocator 대조, 20260930-081655](record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_3gpu_20260930-081655.json) | 64K recording AR의 fragmentation 대조 |
| R35-2 | [35B 2 GPU, 20260929-192934](record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_2gpu_20260929-192934.json) | 용량 한계 참고. 3 GPU와 절대 latency 비교에 사용하지 않음 |
| RB9 | [9B rollback 원본](record_arch_main/rollback/qwen3.5-9b.json), [CSV](record_arch_main/rollback/csv/rollback.csv) | 거절 후 state·logit 불일치의 직접 증거 |
| L9 | [9B 이전 audit](record_arch_main/qwen3.5-9b/sweep_causal_conv1d+fla_20260929-110754.json) | stock/replay 보정 결과·오차 비교 |
| L35 | [35B 이전 audit](record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_20260929-120532.json) | stock/replay 보정 결과·오차 비교 |
| L9T | [9B torch reference audit](record_arch_main/qwen3.5-9b/sweep_torch_reference_20260929-121653.json) | 32K에서 보정 후 AR 일치 사례 |
| L8 | [8B 이전 audit](record_arch_main/qwen3-8b/sweep_causal_conv1d+fla_20260929-104212.json) | recurrent state 결함과 수치 차이를 구분하는 대조군 |

각 sweep의 CSV는 같은 모델 디렉터리의 `csv/<JSON 파일명에서 .json을 뺀 이름>/`에 있다. 메모리는 `component.csv`, latency·speedup은 `timing.csv`와 `speedup.csv`, routing은 `moe_routing.csv`를 사용했다. Trace는 `traces/<timestamp>/*breakdown.json`을 사용했다. 정의는 [MEASUREMENT_OUTPUTS.md](MEASUREMENT_OUTPUTS.md), 측정 결함 수정 이력은 [MEASUREMENT_AUDIT.md](MEASUREMENT_AUDIT.md)를 따른다.

**비교 조건과 제한은 다음과 같다.**

- R8/R9/R35는 RTX A6000 3장, layer sharding, B=1, greedy, 기본 block 16, 출력 256이다. 파일명의 `b16`은 배치가 아니라 **block 폭**이다. Tensor parallelism이나 expert parallelism 실험이 아니므로 TP all-reduce나 EP all-to-all 병목을 관찰했다고 쓰면 안 된다.
- 같은 LongBench 2wikimqa 원문(source index 66)을 길이별로 구성했다. 조건당 프롬프트는 1개이고 성능 패스는 3회 반복한다. 반복 timing의 안정성과 workload 일반성은 별개다.
- R8의 64K는 target/drafter의 설정된 `max_position_embeddings=40960`을 넘는다. R35의 drafter는 기록상 학습 context가 40960이며 64K에는 `beyond_draft_trained_context`가 표시된다. R9의 학습 context는 미확인이다. **64K의 커널·메모리 관찰은 유효하지만, acceptance 저하를 아키텍처의 본질로 해석해서는 안 된다.**
- L8/L9/L35/L9T는 v2와 다른 이전 실험이다. 정확성 결함의 증거로만 사용하며, 그 acceptance를 R9/R35의 latency에 결합해 “보정 speedup”을 만들지 않는다. R35와 R35X의 64K에서는 native AR도 별도 프로세스 사이에서 출력 index 110에 분기했고 stock acceptance가 0.417/0.590으로 달랐다. 커널·프로세스 재현성과 rollback 결함을 별도 요인으로 다룬다.
- R9/R35 v2에는 audit/exact 패스가 없다. 현재 speedup 표는 정확한 복구 비용을 포함하지 않는다. 서로 다른 생성 경로 때문에 stock speedup이 정확한 speedup의 수학적 상한이라고도 단정할 수 없다.
- 메모리 단위는 decimal GB다. 시점별 component 합, 전체 실행의 동시 peak, 장치별 peak의 합은 다른 값이다. 서로 다른 시점의 최대값을 더해 메모리 peak라고 부르지 않는다.

## 3. KV cache 병목은 어디로 이동했는가

| 영역 | Conventional attention에서 중요한 비용 | GDN·MoE에서 드러나는 비용 | 현재 증거의 성격 |
|---|---|---|---|
| 저장 용량 | 토큰별 target K/V, O(S) | Prompt 전체 conv 기록, conditioning feature, 일시적 draft KV | 메모리 계측으로 직접 확인 |
| 정확한 거절 처리 | KV 길이를 수락 경계로 crop | Recurrent state checkpoint·복원·재계산 | 복구 누락 확인. 올바른 복구의 성능은 추가 측정 필요 |
| 짧은 context의 target 실행 | 가중치 읽기를 여러 토큰에 상각 | MoE expert union 증가, 작은 expert GEMM, launch·dispatch | routing과 trace로 확인. 실제 HBM traffic은 미측정 |
| 긴 context의 target 실행 | KV 읽기 및 cache append | 남아 있는 full attention의 masked verify 경로와 복사 | 커널 경로와 시간 확인. 개별 복사 텐서의 정체는 미확인 |
| Request 시작·재사용 | Target prefix cache | Drafter conditioning 재구축 및 hybrid state의 prefix snapshot | cold-start 측정은 있음. prefix-hit sweep은 미완료 |

즉 **용량 병목의 이동**과 **시간 병목의 이동**을 분리해야 한다. 큰 prefill 버퍼가 OOM을 일으키면서도 decode 시간의 대부분은 verify가 차지할 수 있다. 또 작은 recurrent state는 context 길이에 따른 상태 저장을 줄이지만, 짧은 speculative block의 효율적인 실행과 정확한 복구까지 자동으로 제공하지 않는다.

## 4. 병목 A: target KV 절감이 prefill의 speculation 상태에 잠식된다

### 4.1 구체적인 형태

64K에서 직접 계측한 구성 요소는 다음과 같다. [R8/R9/R35, `component.csv`]

| 구성 요소, GB | Qwen3-8B | Qwen3.5-9B | Qwen3.5-35B-A3B |
|---|---:|---:|---:|
| Target attention KV, prefill 끝 | 9.664 | 2.147 | 1.342 |
| Target GDN working state | 0 | 0.0519 | 0.0649 |
| GDN conv recording, prefill 끝 | 0 | **25.768** | **32.210** |
| Prompt context feature | 2.684 | **4.295** | **2.147** |
| Draft KV, 첫 호출 뒤 | 1.343 | 1.611 | 1.611 |
| Draft KV, steady | 1.347 | 0.353 | 0.353 |
| Conv recording, steady | 0 | 0.00629 | 0.00786 |
| 별도 drafter 가중치 | 2.097 | 2.584 | 0.772 |

9B의 conv recording은 target attention KV의 약 **12배**, 35B는 **24배**다. Prompt feature도 각각 2.0배, 1.6배다. Native AR 대비 메모리 overhead는 장치별 peak 합의 차 기준 8B +3.17 GB, 9B +29.43 GB, 35B +33.52 GB다. 이 차를 동시 peak 차라고 해석하면 안 된다. [R8/R9/R35, `speedup.csv`]

따라서 가설 ①은 지지되지만 원인의 우선순위가 바뀐다. **주 병목은 steady drafter cache보다 prefill 동안 불필요하게 보존되는 history와 conditioning의 수명이다.**

### 4.2 원인: committed prompt에도 rollback recording을 적용한다

측정 코드의 [`_make_cache`](dflash/model.py)는 cache 생성 직후 `activate_past_recording()`을 호출한다. 이미 확정된 prompt의 prefill에도 conv 입력 history가 축적된다. 하지만 speculative 거절로 되돌릴 범위는 원칙적으로 현재 verify block이다. Prompt 전체를 되돌릴 필요는 없다.

또한 `crop`이 작은 view만 남겨도 큰 underlying storage가 해제되는 것은 아니다. 실제 `prefill_end`와 `first_draft` component 값에는 25.8/32.2 GB의 recording storage가 계속 남으며, 다음 갱신 이후 steady에서 작아진다. 메모리 분해의 기준은 [`statemem.py`](dflash/arch/statemem.py)의 storage accounting이다.

**이 O(S) conv recording은 GDN의 필연적 메모리 복잡도가 아니다.** 현재 speculation cache의 recording 범위와 allocation 수명에서 생긴 구현 비용이다. Prefill recording을 없애도 recurrent rollback이 해결되지는 않는다. 두 문제를 따로 고쳐야 한다.

### 4.3 Drafter conditioning에 남는 구조적 불일치

Target은 과거 대부분을 고정 크기 recurrent state로 압축하지만 DFlash는 선택된 target hidden을 prompt 길이만큼 materialize하고 drafter에 전달한다. 따라서 target state는 작아져도 conditioning 인터페이스에는 O(S) 비용이 남는다.

9B/35B의 drafter는 sliding attention 5층과 full attention 1층이다. 그래서 초기 KV/target-KV 비율 0.75/1.20이 steady에서는 약 0.164/0.263으로 감소한다. 다만 **full-attention 1층이 남으므로 steady draft KV 전체가 O(1)이 되는 것은 아니다.** 64K의 0.353 GB는 그 길이의 관측값이지, 무한히 긴 context에 대한 고정 상한이 아니다.

첫 draft forward는 64K에서 8B 188 ms, 9B 287 ms, 35B 149 ms다. 이 구간은 projection뿐 아니라 해당 draft forward 전체다. 이를 전부 hidden projection 비용으로 부르면 과도한 귀속이다. 더구나 [`dflash_generate`](dflash/model.py)는 target prefill과 첫 target token 생성 직후 TTFT를 닫고 **그 다음에 첫 draft를 수행한다.** 따라서 첫 draft 시간은 현재 정의에서 TTFT가 아니라 첫 decode 간격과 E2E에 들어간다. [R8/R9/R35 `timing.csv`, `dflash/model.py`]

### 4.4 파생 연구 질문

**RQ-A1. 정확한 rollback을 지원하면서 speculation 전용 history를 O(S)에서 O(γ)로 제한할 수 있는가?**

실험은 native prefill 후 conv의 마지막 window만 독립 저장하고, decode부터 block 범위 recording을 켜는 대조군으로 시작한다. `crop` 후 `storage_bytes`까지 작아지는지 확인해야 한다. 전후 peak·OOM 경계·prefill 시간과 동일-prefix state/logit을 비교한다. 이 수정 자체는 우선 구현 정상화이며, 연구 기여는 이후 정확한 recurrent commit 정책과의 결합에서 찾아야 한다.

**RQ-A2. Target의 작은 상태를 유지하면서 drafter가 긴 context를 충분히 활용할 수 있는가?**

첫 단계는 동일 conditioning을 chunk별로 투영·소비해 prompt feature의 최대 live size를 줄이는 것이다. 이는 peak 감소를 기대할 수 있지만 전체 O(S) 처리량까지 제거하지는 않는다. 다음 단계로 recurrent state 기반 요약, 소수 context feature, 압축된 conditioning을 비교할 수 있다. 이들은 drafter 정확도·학습 변화가 필요할 수 있으므로 메모리 감소와 함께 확정 토큰당 시간을 측정해야 한다.

**RQ-A3. Prefix cache는 target뿐 아니라 speculation에 필요한 상태를 얼마나 재사용해야 하는가?**

Target KV hit만으로 GDN state나 drafter feature가 복원되는 것은 아니다. 같은 prefix의 target KV+recurrent+conv, conditioning feature/draft KV를 어느 수준까지 저장할지 비교한다. 정확한 prefix snapshot과 drafter 버전·position 정보를 cache key에 포함하고, cold/hit 상태의 TTFT·첫 decode 간격·메모리 비용을 측정한다. 현재 데이터로 high-hit 환경의 악화를 입증한 것은 아니다.

## 5. 병목 B: recurrent state의 rollback은 정확성과 비용을 함께 결정한다

### 5.1 단순한 관찰이 아니라 복구 누락의 직접 증거가 있다

RB9는 prompt 512, block 16에서 accepted=0으로 **검사 블록 전체를 취소**한다. 이때 attention KV 16개와 conv state 24개는 segmented reference와 일치하지만 recurrent state 24개는 모두 다르다. Recurrent 최대 절대차는 12.10845, probe logit 최대 절대차는 4.07031이며 probe의 argmax도 다르다. Accepted=0은 이 독립 검사의 조건이며, 실제 DFlash에서 anchor를 포함한 commit 길이 0을 의미하는 것은 아니다. [RB9]

[`rollback.py`](dflash/arch/rollback.py)가 추적하는 측정 구현에서 `crop`은 attention KV와 conv history를 자르지만 recurrent state를 복원하지 않는다. Recurrent update가 in-place이므로 verify 전 상태도 사라진다. 이 결함은 [`test_gdn_recurrent_state_does_not_roll_back`](tests/test_arch.py)의 회귀 테스트에도 명시되어 있다. 실제 설치 라이브러리 전체나 모든 GDN backend가 영구히 그렇다는 주장은 하지 않는다.

이를 식으로 쓰면, verify가 γ개 입력을 처리하고 c개만 commit했을 때

\[
h^{\mathrm{stock}}=h_{t+\gamma},\qquad
h^{\mathrm{correct}}=h_{t+c}
\]

이어야 할 두 상태가 섞인다. c 이후의 거절 토큰이 다음 target 예측에 영향을 미치므로, 이후의 verifier는 의도한 accepted prefix의 target 분포를 계산하지 않는다. **“Target이 verify했으니 lossless”라는 전제 자체가 무너진다.**

### 5.2 기존 replay 감사가 보여 주는 것과 보여 주지 못하는 것

기존 [`VerifyAudit`](dflash/arch/lossless.py)는 verify 전 recurrent/conv state를 저장하고, 수락된 입력만 replay하는 참조를 만든다. 아래는 이전 FLA 감사의 일부다. 분기 위치는 출력의 0-based index이며, 새 v2 timing과 결합하지 않는다.

| 조건 | Stock acceptance → replay 보정 acceptance | Stock와 AR의 첫 분기 | 보정 경로와 AR의 첫 분기 | Reject step의 incremental state 상대오차 중앙값 |
|---|---:|---:|---:|---:|
| 9B, 4K [L9] | 0.429 → 0.228 | 18 | 181 | 0.290 |
| 9B, 64K [L9] | 0.577 → 0.321 | 36 | 241 | 0.358 |
| 35B, 4K [L35] | 0.193 → 0.248 | 42 | 256토큰 전체 일치 | 0.352 |
| 35B, 8K [L35] | 0.445 → 0.414 | 23 | 256토큰 전체 일치 | 0.371 |

이 사례들은 복구가 acceptance를 높일 수도 낮출 수도 있음을 보여 준다. Stock acceptance를 그대로 두고 replay 시간만 더하면 정확한 시스템의 speedup을 예측할 수 없다.

반대로 `exact`라는 패스 이름만으로 AR 출력 일치를 보장할 수도 없다. L9의 FLA 보정 경로는 모든 길이에서 AR과 어딘가 분기한다. 32K의 torch reference 보정은 AR과 256토큰 모두 일치한다[L9T]. Full-attention 대조군 L8에도 일부 출력 분기가 있다. 이는 kernel 폭·연산 순서·정밀도의 영향을 별도로 검증해야 함을 보여 주지만, **보정 후의 모든 잔여 차이가 단순 반올림이라고 증명한 것은 아니다.**

추가로 감사 코드의 `accumulated` 참조는 **shadow recurrent state만** 별도로 유지하고, 과거 attention KV와 conv prefix는 live cache를 재사용한다. Hybrid 모델에서는 이전 오염이 이후 attention KV에도 전파될 수 있으므로 이것을 완전히 독립적인 clean-history oracle로 간주하면 안 된다. Incremental 비교는 같은 step 시작 상태에서 이번 거절의 영향을 분리하는 데 적합하며, 전체 누적 오차의 엄밀한 검증에는 독립된 전체 hybrid cache가 필요하다. [코드 검토에 따른 판단: `lossless.py::_shadow`, `replay`]

### 5.3 왜 단순 crop이나 역연산으로 해결되지 않는가

Attention KV는 token-indexed append이므로 거절 구간을 자르면 된다. Recurrent state는 과거 입력이 중첩된 요약이어서 마지막 상태에 토큰별 삭제 위치가 없다. GDN의 한 가지 행렬 방향 표기로는

\[
H_t=g_t(I-\beta_t k_t k_t^\top)H_{t-1}+\beta_t k_t v_t^\top
\]

처럼 나타낼 수 있다. 이 recurrence의 성질은 [Gated Delta Networks 원 논문](https://arxiv.org/abs/2412.06464)과 일치한다. \(\|k_t\|=1\)인 이상화 아래 역변환에는 \(1/g_t\), \(\beta_t/(1-\beta_t)\)가 들어간다. 강한 forgetting·write 구간에서 작은 수치 오차가 증폭되며, 유한 정밀도에서 잃은 정보를 역산으로 보장해서 복구할 수 없다. [식과 inverse의 로컬 분석: `PROFILING3.md` §5.5.2]

다만 이것은 **정확한 speculative decoding이 원리적으로 불가능하다는 뜻이 아니다.** Verify 이전 또는 중간의 상태를 보존하거나, 올바른 prefix에서 다시 계산하면 된다. 또한 임의의 recurrent 구조가 모두 같은 역조건수를 갖는다고 일반화해서는 안 된다. Mamba로 일반화할 부분은 “토큰별 KV crop 대신 state commit을 설계해야 한다”는 요구다.

### 5.4 해결 방향과 비용

| 방식 | 올바른 수락 경계 상태를 얻는 방법 | 추가 비용과 한계 | 연구에서의 역할 |
|---|---|---|---|
| Verify 전 snapshot + accepted prefix 전체 target replay | Recurrent/conv 복원, KV crop 후 commit된 입력 재실행 | 가장 단순한 참조 구현. Attention·MoE까지 중복 실행하며 성능 손해가 큼 | 정확성 oracle, 비용의 기준선 |
| Token별 recurrent state snapshot | Verify kernel이 각 위치의 상태를 저장하고 수락 위치를 선택 | 추가 저장·write traffic O(γR). Conv window와 KV도 같은 경계로 정렬 | 구현 가능한 기본 성능 대조군 |
| Recurrent update 입력 tape + state-only replay | 올바른 시작 상태와 token별 update 계수로 GDN recurrence만 재실행 | O(γD) 수준의 tape와 replay 비용. 시작 상태가 정확하고 각 layer의 causal update 입력이 보존돼야 함 | 전체 attention/MoE replay를 피하는 후보 |
| 희소 checkpoint + 국소 replay | 몇 개 경계의 snapshot과 그 사이 tape를 보존 | Snapshot traffic과 replay 거리의 trade-off | 수락 경계 분포에 따른 최적 정책 연구 |

여기서 R은 전체 recurrent state 크기, D는 token별 update 입력 저장량이다. 현재 working state 크기를 보수적인 R 근사로 사용하면 γ=16의 추가 snapshot 용량은 9B 약 0.83 GB, 35B 약 1.04 GB다. Conv working state가 포함된 근사이며 실제 recurrent-only 계산은 조금 작다. **이는 측정 성능이 아닌 용량 추정**이다. Prompt recording 25.8/32.2 GB보다 작지만 매 verify마다 state를 쓰는 bandwidth 비용이 새 병목이 될 수 있다. 시작 checkpoint까지 추가 저장하면 그만큼 더 필요하다.

Token별 snapshot은 이미 알려진 방법이다. NVIDIA Megatron Core는 gated-delta 계열 kernel의 `intermediate_states`를 수락 경계 복원용으로 설명하며 Mamba2의 intermediate SSM state와의 대응도 명시한다. [Megatron Core 공식 문서](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.ssm.ops.gdp.fused_recurrent.html) 또한 Mamba speculative decoding에서 중간 state 저장과 multi-step kernel을 다룬 선행연구가 있다. [The Mamba in the Llama, §4](https://arxiv.org/html/2408.15237v1)

따라서 **“snapshot을 도입한다”만으로 신규성을 주장하기보다, DFlash conditioning·GDN state commit·MoE 재실행 회피를 함께 고려했을 때 어떤 저장/재계산 정책이 최적인가**를 연구 질문으로 삼아야 한다.

### 5.5 정확성 검증 절차

1. **State 단위 검사:** 동일한 prefix와 시작 cache를 고정한다. γ∈{4,8,16}, 실제 commit 길이 c∈{1,…,γ} 전체에서 verify→commit 결과를 clean prefix 실행과 비교한다. 별도 rollback primitive는 c=0도 검사한다. KV, conv, recurrent, position/length metadata를 모두 확인한다.
2. **수치 차이 대조:** full accept, 거절 없음, dense control, sequential/chunk kernel, FP32 reference를 나눠 비교한다. Full-accept 오차가 0이라는 사실만으로 짧은 replay 폭의 numerical floor까지 0이라고 가정하지 않는다.
3. **동일-prefix logit 검사:** State 상대 Frobenius 오차뿐 아니라 KL, 최대 logit 차이, top-1 margin, argmax flip을 기록한다. EOS를 포함한 첫 출력 분기 시점과 최초 상태 불일치 시점을 분리한다.
4. **독립된 전체-cache shadow:** AR 또는 clean teacher-forced token열을 따라 독립 cache를 갱신한다. Shadow recurrent만 live attention cache에 끼워 넣는 방식으로 clean history를 대체하지 않는다.
5. **성능 재측정:** 검사에 통과한 production commit 경로만 timing한다. Audit의 비교·추가 probe 비용은 제외하되, 실제 snapshot/tape/replay 비용은 반드시 포함한다. 동일 조건에서 acceptance와 routing도 다시 수집한다.

Greedy token 일치, 같은 계산 경로에서의 bitwise state 복원, 수학적 target 분포 보존은 서로 다른 주장이다. 향후 sampling으로 확장하면 올바른 rejection sampling과 cache 복구를 함께 검증해야 하며, 같은 seed의 문장 일치만으로 분포 보존을 대신할 수 없다.

## 6. 병목 C: MoE verification은 expert당 연산이 커지기보다 expert 집합이 넓어진다

### 6.1 구체적인 형태

35B의 S=4K, block 16에서 verify 1회 GPU kernel 시간 합은 약 54.4 ms다. Routed expert GEMM이 30.58 ms(약 56%), route 1.26 ms, dispatch 2.11 ms, GDN mixer 약 8.01 ms, full attention 약 6.51 ms다. Perf verify 구간은 183.4 ms이고 총 kernel launch는 약 9,239개다. AR은 131.7 ms/step, GPU kernel 합 약 19.1 ms다. [R35, 4K verify/AR breakdown]

단, **54.4/183.4를 정확한 GPU utilization으로, 그 나머지를 모두 Python launch 시간으로 부를 수는 없다.** 서로 다른 perf/trace 패스, 다른 step 범위, multi-GPU 시간 합이며 profiler가 host 실행에 영향을 준다. 반복적인 expert 실행 코드와 많은 launch는 host/launch 병목을 강하게 시사하지만, CPU gap·동기화·P2P·장치 간 대기의 정확한 분해는 추가 timeline 분석이 필요하다.

4K에서 verify는 layer당 평균 48.48개 expert를 사용하고 AR은 8개다. 32K block sweep을 원본 routing CSV에서 재집계하면 다음과 같다. 각 수치는 **layer×step 행의 산술평균**이며 마지막 짧은 block도 포함된다. [R35/R35B]

| Block 폭 | 평균 unique experts/layer | Hit expert당 평균 token 수 | 평균 load imbalance | Verify/회 | 확정 token/회 | TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 21.60 | 1.515 | 2.220 | 195.0 ms | 3.148 | 65.06 ms |
| 8 | 35.57 | 1.860 | 2.999 | 204.7 ms | 5.543 | 38.83 ms |
| 16 | 50.39 | 2.601 | 4.036 | 214.5 ms | 6.375 | 35.37 ms |

Block 4→16에서 expert union은 2.33배지만 hit expert당 평균 token 수는 1.52→2.60에 그친다. **많은 expert의 매우 작은 GEMM을 실행한다**는 해석이 compute saturation보다 먼저다. Load imbalance는 hit expert 사이의 token 수 불균형이며, 현재 expert별 순차 실행에서 이를 곧바로 분산 straggler 시간으로 해석하지 않는다.

### 6.2 원인: token 병렬성이 expert 내 재사용으로 모두 이어지지 않는다

한 layer의 expert 수를 E, top-k를 r=8, verify token 수를 q, expert union을 U(q)라고 하자. 단순한 실행 비용 모델은

\[
T_{\mathrm{MoE}}(q)\approx T_{\mathrm{route/dispatch}}
+T_{\mathrm{launch}}(U)
+T_{\mathrm{expert}}\bigl(\{n_e\}_{e\in U}\bigr)
\]

이다. Expert weight read의 논리적 working set은 대략 \(U(q)W_e\), 연산량은 \(rqF_e\)에 비례한다. 그러므로 expert 내 평균 재사용을 나타내는 \(rq/U(q)\)가 작으면 q 증가가 충분히 큰 GEMM을 만들지 못한다. 실제 HBM 읽기는 L2 재사용·layout·fusion에 따라 달라지므로 U를 그대로 HBM bytes로 치환해서는 안 된다.

원본의 expert GEMM 7.2배 증가도 “compute-bound가 됐다”의 증거가 아니다. GEMM kernel도 weight bandwidth, 작은 shape, 낮은 occupancy에 묶일 수 있다. Routing top-k 자체의 1.26 ms만 줄이는 최적화는 전체 verify에 대한 효과가 제한적이다.

### 6.3 거절 낭비는 token 비율과 expert 비용 비율이 다르다

현재 기록에는 per-token expert ID가 있으나 CSV의 union 요약만으로 rejected compute를 산출할 수는 없다. 다음을 **같은 run의 실제 commit 경계**와 연결해야 한다.

\[
U_{\mathrm{reject-only}}=
\left|\bigcup_{j=c+1}^{q}E_j\;\setminus\;\bigcup_{j=1}^{c}E_j\right|.
\]

Rejected-only expert는 수락 토큰만 검증했더라면 불필요했을 weight working set과 launch를 나타낸다. 수락 토큰과 expert를 공유하는 거절 토큰도 추가 MAC·activation 비용은 발생하지만, 동일한 추가 weight loading 비용을 부과하면 과대계산할 수 있다. 반대로 \((q-c)/q\)에 전체 verify latency를 곱하면 고정비와 kernel shape 변화를 놓친다. 실제 절감 시간은 동일 prefix/candidate를 고정한 accepted-prefix-only 대조로 측정해야 한다.

### 6.4 파생 연구 질문

**RQ-C1. Parallel drafting에서 생기는 expert union과 작은 GEMM의 비용을, 같은 target 계산을 유지하면서 줄일 수 있는가?**

HF expert loop와 grouped/fused execution을 같은 후보 token·routing·state에서 비교한다. Layer별 U, \(n_e\) histogram, weight traffic, launch 수, kernel 시간, end-to-end TPOT를 함께 측정한다. Target router를 임의 변경하거나 expert를 생략하는 방식은 lossless acceleration 대조군에 포함하지 않는다.

**RQ-C2. Accepted-prefix probability가 같아도 expert union과 state commit 비용이 다르면 최적 verification 실행이 달라지는가?**

새로운 변수는 단순한 verify 길이뿐 아니라 **같은 길이를 어떤 expert grouping·recurrent checkpoint 정책으로 실행하는가**다. Router 정보를 얻으려면 선행 layer 계산이 필요할 수 있으므로, 비용을 예측하는 metadata나 predictor의 overhead도 포함해야 한다. Oracle routing으로 얻은 이득을 무료 online 이득으로 제시하지 않는다.

**RQ-C3. MoE를 효율화하면 DFlash의 절대 latency와 상대 speedup이 어떻게 달라지는가?**

현재 AR 자체에 큰 host 비용이 있다. 이를 최적화하면 DFlash도 빨라지지만 AR이 더 큰 비율로 빨라져 상대 speedup은 감소할 수 있다. Stock 35B의 3–4배 speedup을 아키텍처 고유의 이득으로 해석하지 않고, 최적화된 AR과 DFlash를 함께 비교해야 한다.

## 7. 병목 D: 작은 target KV에도 residual attention verification 비용이 남는다

### 7.1 직접 관찰

| 모델 | Verify/AR step 비용 비율, 4K→64K | 긴 context에서의 관찰 |
|---|---:|---|
| 8B | 1.27→4.48 | 64K verify에서 mem-efficient attention 약 119 ms, elementwise copy 약 110 ms |
| 9B | 1.41→2.41 | Full-attention verify 구성 비용 약 6.1→70.5 ms, GDN 약 8 ms로 거의 일정 |
| 35B | 1.39→1.95 | Full-attention mixer 약 6.5→83.2 ms, GDN 약 8 ms로 거의 일정 |

출처: R8/R9/R35 및 각 trace breakdown. 8B 64K에는 §2의 context 범위 제한이 적용된다.

35B에서 S=4K의 GPU 시간 최대 항목은 expert GEMM이지만 S=64K에서는 attention이 GPU kernel 시간 합의 약 63%다. **Full-attention 층 수가 적다는 사실은 KV 저장 용량을 줄이지만, 비효율적인 verify 경로가 없어졌다는 뜻은 아니다.**

측정된 verify 경로는 `[1,1,q,S]` 형태의 bool mask와 mem-efficient attention을 사용한다. AR은 다른 flash 경로를 사용한다. 다만 bool mask 자체가 모든 backend에서 반드시 fallback을 일으킨다는 보편 명제는 아니다. 측정한 dtype·shape·mask 표현·backend 조합에서 관찰된 선택이다. Elementwise copy가 mask 확장인지 K/V contiguous 변환인지까지 확정되지는 않았다.

8B의 nominal KV bytes/attention 시간으로 계산한 81 GB/s를 실제 HBM bandwidth로 해석해서도 안 된다. Multi-query verify는 KV를 여러 번 읽거나 변환할 수 있다. 이 값은 논리적인 한 번의 KV 읽기에 대한 유효율이며, 낮다고 해서 bandwidth 제한 가능성을 배제할 수 없다. 기존 보고서의 “대역폭 한계가 아니라 kernel 선택의 결과”는 **“AR과 다른 kernel 경로에 큰 비용이 관찰되며 그 경로의 bandwidth/compute 제한은 아직 미분리”**로 좁혀 쓰는 것이 정확하다.

### 7.2 파생 연구 질문

**RQ-D1. 수락 후보 전체를 그대로 verify하면서, cached prefix와 짧은 causal block에 맞는 attention 경로로 비용을 줄일 수 있는가?**

올바른 offset causal semantics를 지원하는 mask 표현/attention kernel, 장치별 mask 재사용, 정적 또는 paged KV append를 각각 ablation한다. 기존 DynamicCache의 `cat` 비용은 AR에도 존재하므로 양쪽에 같은 최적화를 적용한다. 직사각형 query/key의 causal alignment는 주의가 필요하며 단순히 mask를 지우거나 `is_causal=True`로 바꾸는 것은 정확성 해결책이 아니다.

**RQ-D2. GDN:dense-attention 비율과 recurrent commit 정책을 함께 고려할 때 병목 교차점 S*는 어디인가?**

S와 γ를 바꾸며 attention, recurrent update/commit, FFN/MoE, conditioning 비용을 비교한다. 세 모델 간 절대 latency 차이만으로 층 비율의 인과 효과를 추정하지 말고 모델 내부의 경로 최적화 전후·layer breakdown으로 검증한다. Pure Mamba에는 residual full-attention 문제가 없지만 recurrent commit 문제는 남는다.

## 8. 목표 ③의 재정의: compute-bound 전환을 무엇으로 입증할 것인가

현재 실험은 **S 증가에 따라 attention GPU 비용이 커지고, MoE에서는 expert union과 작은 kernel 실행 비용이 증가한다**는 사실을 보여 준다. Compute-bound 전환은 별도 가설로 남겨야 한다.

| 후보 병목 | 필요한 증거 | 현재 상태 |
|---|---|---|
| Host/launch | CPU launch gap, synchronization, GPU idle, fused/graph 실행 전후 | 많은 launch와 실행 코드로 강한 정황. 정확한 시간 비율은 미확정 |
| HBM bandwidth | 실측 DRAM bytes/throughput, memory stall, cache hit | 하드웨어 카운터 없음 |
| Compute | kernel별 FLOPs, Tensor Core/SM 지표, arithmetic intensity와 empirical roofline | 미측정. GEMM 비중만으로 확정 불가 |
| Recurrent dependency/작은 shape | 상태 update·scan의 critical path, occupancy, width별 latency | GDN의 S 독립성은 관측. γ별 효율 원인은 추가 분해 필요 |
| 통신 | critical path상의 P2P bytes/time | Layer sharding의 일부 전송은 계측. TP/EP 통신은 대상 아님 |

B=1을 유지하고 S∈{4K,8K,16K,32K,64K}, γ∈{4,8,16}에서 검사한다. 상태 복구와 attention/MoE 구현을 정상화한 뒤 다시 측정해야 한다. 카운터를 사용할 수 없으면 roofline은 추정으로 표시하고, bound의 전환점을 확정적으로 보고하지 않는다.

구체적인 분석 모델은 다음과 같다.

\[
T_{\mathrm{iter}}=
T_{\mathrm{draft}}+T_{\mathrm{verify}}
+T_{\mathrm{commit}}+T_{\mathrm{bookkeeping}},\qquad
\mathrm{TPOT}_{\mathrm{exact}}\approx
\frac{\sum_i T_{\mathrm{iter},i}}{\sum_i c_i}.
\]

`commit`에는 checkpoint 저장·state 복원·필수 replay가 포함된다. 검증 kernel에 snapshot 저장을 융합했다면 verify 시간에 포함시키고 중복 합산하지 않는다. 이번 구현의 block 크기 γ는 anchor 1개와 draft γ−1개이며, c는 cache에 반영되는 anchor+accepted prefix의 길이다. 출력 token 수·bonus·마지막 짧은 block까지 고려해 실제 기록의 `committed_per_step`을 사용한다.

AR보다 빨라지려면 대략 \(\mathbb E[c]>T_{\mathrm{iter}}/T_{\mathrm{ARstep}}\)이어야 한다. 현재 stock의 64K 기록에서 decode 전체를 step 수로 나눈 비용을 사용하면 다음과 같다. 첫 draft 비용도 상각되어 포함된다.

| 모델 | 관측 평균 c | AR를 이기기 위한 비용상 c 임계값 | Stock decode speedup |
|---|---:|---:|---:|
| 8B | 1.275 | 4.721 | 0.270× |
| 9B | 9.808 | 2.978 | 3.293× |
| 35B | 7.083 | 2.055 | 3.446× |

출처: R8/R9/R35 perf의 `decode_latency_s / num_verify_steps / ar.time_per_output_token_s` 재계산. 이 표는 stock의 관측 손익분기 설명이며 정확한 복구 이후의 예측치가 아니다.

32K에서 γ=4→16의 verify 비용 증가는 8B 약 3%, 9B 약 5%, 35B 약 10%다. 반면 35B의 c는 약 2배가 된다. 그러므로 현재 B=1 범위에서는 **verify 길이를 줄이면 언제나 빨라진다는 가설도 성립하지 않는다.** 거절 token 비율만 낮추는 것과 확정 token당 비용을 낮추는 것을 구분해야 한다.

## 9. Dspark와 구분되는 연구 질문 및 우선순위

DSpark는 parallel backbone과 lightweight sequential module로 intra-block 의존성을 보완하고, prefix survival probability와 engine throughput profile에 따라 verification 길이를 선택한다. 따라서 “소형 AR conditioner”, “confidence로 suffix verify 생략”, 단순한 “hardware-aware adaptive γ”를 독립적인 신규 기여로 제안하지 않는다. [DSpark 원 논문](https://arxiv.org/abs/2607.05147)

아래 질문은 DSpark의 길이 선택을 대조군 또는 결합 요소로 두고도 검증할 수 있다. 신규성은 후보이며, 각 주제의 추가 선행연구 검토가 필요하다.

| 우선순위 | 연구 질문 | 현재 데이터와의 연결 | 핵심 실험과 반증 조건 |
|---|---|---|---|
| **P0** | **정확한 hybrid state commit의 저장/재계산 비용을 최소화할 수 있는가?** | Recurrent 복구 누락과 25.8/32.2 GB prompt recording이 동시에 존재 | Full replay, token snapshot, tape-only replay, sparse checkpoint 비교. 동등 정확성에서 TPOT/peak Pareto 개선이 없으면 제안의 이점 기각 |
| **P1** | **Target KV가 작아질 때 conditioning 인터페이스도 함께 작게 만들 수 있는가?** | 9B prompt feature 4.30 GB, 첫 draft 287 ms | 동일 feature streaming 후, 압축/state conditioning 비교. acceptance 손실을 포함한 확정 token당 비용이 나빠지면 압축안 기각 |
| **P1** | **MoE verification의 expert 집합 확장을 실제 reuse로 바꿀 수 있는가?** | γ=16에서도 expert당 평균 2.6 token | 동일 token/routing의 fused/grouped 경로와 baseline 비교. 최적화 후 남는 U·shape별 비용과 bandwidth/compute 제한 분석 |
| **P2** | **Verification 길이와 별도로 commit checkpoint·expert 실행 형태를 공동 선택하면 이득이 있는가?** | 같은 γ라도 state traffic·U·accept 경계가 비용을 결정 | DSpark 길이 선택만, commit 정책만, 실행 정책만, 공동 정책의 ablation. 예측 overhead까지 포함해 추가 이득이 없으면 기각 |
| **P2** | **Long-context hybrid에서 정확한 state commit과 residual attention 중 어느 쪽이 최종 병목인가?** | 35B의 4K expert 중심→64K attention 중심 변화 | Correctness 확보 후 attention 경로를 최적화하고 S 교차점 재측정. 최적화 후 현상이 사라지면 backend-specific 현상으로 보고 |

가장 설득력 있는 중심 주제는 **“정확한 상태 확정을 포함한 DFlash의 아키텍처별 비용”**이다. DSpark가 “얼마나 멀리 verify할 것인가”를 다룬다면, 여기서는 **선택된 길이를 검증한 뒤 어떤 상태를 얼마의 비용으로 확정하고, sparse expert 실행을 어떻게 유지할 것인가**를 묻는다. DSpark에도 engine 비용 모델이 있으므로 단순히 비용 항을 추가하는 것만으로 차별화되지는 않는다. 새로운 state 표현·복구 kernel·실행 정책과 실증이 필요하다.

## 10. 배치를 제외한 후속 실험 계획

모든 단계는 B=1로 진행하며, 현재 브랜치의 범위를 유지한다.

| 순서 | 실험 | 통제할 것 | 반드시 보고할 결과 |
|---|---|---|---|
| 1 | Rollback correctness와 clean cache oracle | 동일 token prefix, 동일 backend/precision, 강제 commit 경계 | State·logit 오차, AR 첫 분기, full-accept numerical control |
| 2 | Prefill recording 제거와 storage 수명 단축 | 동일 drafter/target, conditioning 내용 유지 | 구간별 live/storage/peak bytes, OOM, TTFT와 첫 decode 간격 |
| 3 | Exact commit 전략 비교 | S·γ·후보 token·수락 경계 고정 | Snapshot/tape bytes, replay FLOPs/시간, exact TPOT |
| 4 | Attention 경로와 MoE 실행 정상화 | 동일 state·candidate, AR에도 같은 최적화 적용 | Module/kernel 시간, launch·P2P, U와 expert shape |
| 5 | S·γ 공동 sweep | 기본 출력 256, 여러 prompt와 process 반복 | Exact speedup, committed token/step, measured/estimated bound 구분 |
| 6 | 출력 256/1K/4K와 prefix hit 0/50/90% | 같은 재사용 prefix와 suffix, state/cache policy 명시 | TTFT, 첫 decode 간격, steady TPOT, E2E, 추가 cache 용량 |

5단계까지는 실제 발생하는 비용의 원인을 식별하는 실험과 end-to-end generation을 함께 사용한다. 고정 후보·강제 수락 경계 실험은 비용을 분리하는 microbenchmark일 뿐, 실제 acceptance 기반 speedup의 대체물이 아니다.

6단계에서는 setup 비용의 상각을 다음과 같이 모델링할 수 있다.

\[
T_{\mathrm{DF}}\approx P_{\mathrm{DF}}+F_{\mathrm{setup}}+(O-1)\tau_{\mathrm{DF}},
\quad
T_{\mathrm{AR}}\approx P_{\mathrm{AR}}+(O-1)\tau_{\mathrm{AR}}.
\]

\(\tau_{\mathrm{AR}}>\tau_{\mathrm{DF}}\)일 때만
\(O-1>(P_{\mathrm{DF}}-P_{\mathrm{AR}}+F_{\mathrm{setup}})/(\tau_{\mathrm{AR}}-\tau_{\mathrm{DF}})\)
형태의 상각 경계를 논할 수 있다. 여기의 steady τ에서는 first-draft setup을 제외해 중복 계산하지 않는다. 현재 O=256 실험만으로 O=1K/4K 또는 prefix-hit 효과를 관측했다고 쓰지 않는다.

최종 결과표에는 기존 KV overhead ratio 외에 다음 지표를 추가한다.

- **메모리:** 시점별 draft KV/target attention KV, rollback 전용 bytes/target working state, peak/steady 분리, 장치별 용량 여유와 allocator fragmentation.
- **시간:** first-draft setup, verify mixer/FFN, exact commit, host·통신 비용, 확정 token당 전체 시간.
- **MoE:** U(q), expert별 token histogram, rejected-only expert union, grouped execution 이후 kernel shape별 효율.
- **정확성:** state/logit 오차, AR와의 첫 출력 분기, numerical floor 대조, 정확한 경로에서 재측정한 acceptance.

## 11. 기존 보고서와 실험 계획을 인용할 때의 표현 정리

| 피해야 할 단정·혼동 | 이 보고서의 보완 |
|---|---|
| “GDN rollback은 lossless가 아니다” | **현재 측정된 crop 구현이 recurrent state를 복원하지 않는다.** GDN 자체의 lossless speculation 불가능성을 뜻하지 않는다 |
| “작은 KV 대신 drafter overhead가 지배한다” | 가장 큰 메모리 항목은 prompt conv recording과 context feature다. Steady draft KV·가중치는 별도 항목이다 |
| “Sliding drafter 때문에 steady cache는 제한된다” | 5개 sliding층은 제한되지만 full-attention 1층의 O(S) 항은 남는다 |
| “First draft projection이 TTFT를 늘린다” | 현재 코드에서 first draft는 TTFT 종료 이후다. First-draft 전체 시간과 projection 단독 시간도 구분한다 |
| “35B verify 시간 중 나머지는 모두 host launch” | Host/launch의 강한 정황이지만 trace 합과 별도 perf 구간의 차를 정확한 CPU 시간으로 볼 수 없다 |
| “낮은 유효 GB/s이므로 bandwidth-bound가 아니다” | Logical KV bytes/time은 실제 DRAM throughput이 아니다. 비효율적 attention 경로의 내부 병목은 미확정이다 |
| “MoE가 compute-bound 전환을 보인다” | Expert union·GEMM·launch 증가는 확인되지만 compute 포화는 미입증이다 |
| “64K acceptance 차이는 아키텍처 효과다” | 8B RoPE 범위, 35B drafter 학습 길이, 단일 prompt, rollback 오염의 영향을 함께 통제해야 한다 |

이 보완을 적용하면 현재 자료는 **“KV 절감 이후 드러나는 speculation 상태 관리와 sparse verification의 비용”**을 진단하는 근거로 충분하다. 다음 연구 결과의 성패는 정확한 state commit을 확보한 뒤에도 그 비용 이동과 제안한 최적화의 이득이 유지되는지에 달려 있다.
