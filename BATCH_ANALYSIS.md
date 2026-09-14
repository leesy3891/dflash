# DFlash 배치 스윕 분석 보고서 (fla 재측정 포함)

작성 2026-09-14. 대상 기록은 두 가지다.

| 디렉터리 | 모델 | GDN kernel | 측정일 |
| --- | --- | --- | --- |
| `record_batch/` | Qwen3-8B, Qwen3.5-9B | torch fallback (fla 없음) | 2026-09-10 |
| `record_batch_fla/` | Qwen3.5-9B | fla 0.5.2 (`chunk`, `fused_recurrent`) | 2026-09-12 |

설계와 지표 정의는 [BATCH_PROFILING_PLAN.md](BATCH_PROFILING_PLAN.md)에 있다. 그림은
`visualization_batch/`에 있고, fla 스윕의 그림은 `_fla` 접미사가 붙는다. 이 문서의 모든
수치는 두 디렉터리의 기록과 `visualization_batch/*.csv`에서 왔다. 추정치는 **(추정)**으로
표시했다.

본문에서는 표기를 줄여 쓴다. **8B**는 Qwen3-8B, **9B**는 fla 스윕의 Qwen3.5-9B,
**9B-fb**는 fallback 스윕의 Qwen3.5-9B다. BL은 baseline(한 step에 B×1 토큰), DF는
DFlash(한 step에 B×16 토큰)다.

---

## 0. 요약

**속도: 주된 병목과 발생 위치**

- **작은 B(≤8)**에서는 DF step이 BL step보다 8B는 1.36–1.60배, 9B는 2.08–2.18배 느리다.
  이 차이는 target verify가 아니라 **DFlash가 추가하는 일**이 만든다.
  - 8B: draft forward와 LM head가 BL step의 0.27배다.
  - 9B: GDN rollback 0.48배, draft 0.30배, verify 초과분 약 0.3배다.
  - 16토큰 verify 자체는 memory-bound라 거의 공짜다(c_verify 1.09–1.4).
- **큰 B(≥16, 4k 기준)**에서는 verify가 **compute-bound로 넘어간다.** DF verify는 BL verify의
  2–3배가 되고, 검증한 16토큰 중 τ≈3.4–4.2개만 채택되므로 나머지 연산은 버려진다.
  - static batch의 꼬리 비용이 여기에 곱해진다. 끝난 행까지 verify하므로 행-step당 유효
    토큰이 τ의 57–72%로 줄어든다.
  - 그 결과 same-B speedup이 B=16 부근에서 1을 밑돈다.
- 위치로 말하면 decode step 안의 세 곳과 스케줄링 한 곳이다.
  1. `target(**kwargs)` verify forward (`dflash/batch.py:800`)
  2. `tcache.commit` GDN replay (`:823`)
  3. `self.draft` + `compute_logits` (`:876`, `:886`)
  4. 끝난 행을 끝까지 끌고 가는 static batch

**메모리: DFlash가 추가하는 것**

- B·S에 비례하는 항은 **draft KV 하나뿐이다**(8B 20 KiB/token = target KV의 14%, 9B 24 KiB/token =
  75%).
- **Tapped hidden state**는 S에 비례하지만 한 행 분량이라 B와 무관하다(8B 40 KiB/token, 9B 64 KiB/token).
  target prefill 도중에 생기므로 **target prefill 피크 안에 포함되어 보인다.**
- DF−BL 피크 차이는 **draft KV + tapped hidden 한 행분**으로 설명된다. 8B는 전 점에서 0.1 GiB,
  9B는 0.25 GiB 이내다.
- 9B는 여기에 **decode 중 행당 약 35 MiB의 rollback scratch**가 더해진다. S가 작고 B가 크면 이것이
  피크가 된다.
- 9B에서 DFlash만 OOM이 나는 두 점(16k×32, 32k×16)은 **draft sliding layer를 전체 길이로 저장한
  구현** 때문이다. ring buffer를 쓰면 fla 기준으로 둘 다 들어간다 (추정).

**fla 전후 (9B)**

- 작은 B의 speedup은 크게 올랐다(4k B=1 1.50 → 1.97).
- 큰 B는 오히려 내려갔다(4k B=32 0.93 → 0.87, 8k B=32 0.73 → 0.67). fallback 시절 baseline이
  kernel-bound였는데, fla로 대역폭 한계(copy BW의 71–73%)까지 따라붙었기 때문이다.
- τ, 출력 일치율, OOM 경계는 변하지 않았다. target prefill의 activation transient는 217 → 141
  KiB/token으로 줄었다.

**경향**

- **S가 길어질수록** B=1 speedup이 줄어든다.
  - 8B는 크게 줄어든다(2.49 → 1.05). τ 하락, 16토큰 attention 비용 증가, 무거운 target KV가
    원인이다.
  - 9B는 완만하게 줄어든다(1.97 → 1.68). attention 층이 8개뿐이다.
  - max B는 S가 두 배가 될 때마다 절반이 된다.
- **B가 커질수록** speedup이 줄어든다.
  - τ는 B와 무관하다.
  - step 비용 비율은 B≈8까지 평탄하다가 B=16에서 뛴다(compute-bound 전이).
  - 꼬리 비용이 누적된다.
  - DFlash 추가 메모리가 B에 선형으로 늘어난다.
  - iso-memory speedup은 8B 16k(1.01)를 빼면 전부 1 미만이다.

---

## 1. 실험 구성과 읽는 법

- **격자.** S ∈ {4k, 8k, 16k, 32k}, B ∈ {1, 2, 4, 8, 16, 32}, 32 prompts, 출력 N=256 고정(EOS 억제),
  greedy, 블록 16.
- **엔진.**
  - 행별 static KV를 setup에서 모든 행분 미리 할당한다.
  - prefill은 요청 단위로 순차 진행한다.
  - decode는 B행을 한꺼번에 처리한다.
  - GDN rollback은 채택된 prefix를 replay한다.
  - 기기는 RTX A6000 한 장(47.43 GiB)이다.
- **Throughput.** makespan은 전체 토큰 수 ÷ decode 시간이다. full-occupancy는 모든 행이 살아
  있던 step만 센다.
- **Speedup 항등식.** 모든 점에서 기록값과 일치한다.

  ```
  makespan speedup = (행-step당 유효 토큰) / (DF step 시간 / BL step 시간)
  행-step당 유효 토큰 = DF step당 생산 토큰 / B      (끝난 행 포함; τ보다 작다)
  ```

  전체 τ로 나누면 과대평가된다. EOS 이후 구간은 τ가 더 높고, 끝난 행도 계속 verify되기 때문이다.
- **Bound 판정** (plan §5.4). copy 대역폭은 실측 720 GB/s, GEMM 최대 성능은 132 TFLOP/s이고
  ridge는 183 FLOP/byte다. copy BW의 70% 이상이면 HBM-bound, GEMM 최대 성능의 60% 이상이면
  compute-bound다. byte와 FLOP은 이상적 kernel을 가정한 analytic 값이다.

---

## 2. 속도: 주된 병목은 무엇이고 어디서 느려지는가

### 2.1 Step 비용 분해

DF step을 BL step으로 나눈 값의 구성이다. 모든 값은 BL step 대비 배수이며, 나머지 ctx와 cpu는
0.01 미만이다.

| 점 | DF verify | draft + LM head | accept/rollback | DF step / BL step | 행-step당 토큰 | speedup (makespan / full) |
| --- | --- | --- | --- | --- | --- | --- |
| 8B 4k B=1 | 1.09 | 0.27 | 0.00 | 1.36 | 3.39 | 2.49 / 2.49 |
| 8B 4k B=8 | 1.28 | 0.28 | 0.01 | 1.57 | 2.54 | 1.62 / 2.07 |
| 8B 4k B=16 | 1.97 | 0.37 | 0.01 | 2.35 | 2.46 | 1.05 / 1.32 |
| 8B 4k B=32 | 2.24 | 0.43 | 0.01 | 2.68 | 2.28 | 0.85 / 1.11 |
| 8B 32k B=1 | 1.60 | 0.24 | 0.00 | 1.85 | 1.93 | 1.05 / 1.05 |
| 9B 4k B=1 | 1.26 | 0.30 | 0.53 | 2.08 | 4.10 | 1.97 / 1.97 |
| 9B 4k B=8 | 1.26 | 0.36 | 0.53 | 2.14 | 3.24 | 1.51 / 1.98 |
| 9B 4k B=16 | 1.85 | 0.48 | 0.46 | 2.79 | 2.95 | 1.06 / 1.51 |
| 9B 4k B=32 | 2.39 | 0.64 | 0.39 | 3.42 | 2.97 | 0.87 / 1.21 |
| 9B 32k B=8 | 1.38 | 0.36 | 0.44 | 2.18 | 2.82 | 1.29 / 1.74 |

(9B의 accept/rollback 열에는 BL도 내는 commit 비용이 포함된다. 9B 4k B=1에서 DFlash만의 추가분은
0.48배다.)

**작은 B.** 비율이 B와 거의 무관하다(8B 1.36–1.57, 9B 2.08–2.18). 이 구간의 비용은 **DFlash가
따로 하는 일**이다.
- 8B: draft forward와 LM head
- 9B: GDN rollback(약 20 ms)과 draft(약 11 ms)
- verify 초과분(DF verify ÷ BL verify − 1)은 4k 기준 8B 9–28%, 9B 33–41%에 그친다.

이때 speedup의 천장은 τ다. 8B는 τ 3.45에 비율 1.36을 곱해 2.49, 9B는 4.10에 2.08을 곱해 1.97이다.

**B=16에서 전이가 일어난다.**
- DF verify: 8B 1.28 → 1.97, 9B 1.26 → 1.85
- draft도 BL step 대비 커진다: 8B 0.27 → 0.43, 9B 0.30 → 0.64
- 반면 rollback 비중은 줄어든다. B와 무관한 고정 비용이기 때문이다.

### 2.2 Memory-bound에서 compute-bound로 (`batch_bound_*`)

| | BL verify, copy BW 활용률 | DF verify, BW / GEMM 활용률 | DF verify intensity (ridge 183) |
| --- | --- | --- | --- |
| 8B 4k | 75–81% (HBM-bound) | B=1 69%/7% → B=16 42%/40% → B=32 35%/48% | 18 → 178 → 255 |
| 9B-fb 4k | 49–63% (kernel-bound) | 31%/3% → 29%/33% → 20%/39% | 16 → 211 → 350 |
| 9B 4k | 64–73% (HBM-bound, B≥4) | 48%/4% → 33%/39% → 24%/46% | 16 → 211 → 350 |

- **Baseline(q=1)**는 모든 B에서 memory-bound다. 따라서 B를 늘리는 만큼 거의 공짜로 throughput이
  는다.
- **DF verify(q=16)**의 intensity는 약 16B로 오르고, B≈16에서 ridge를 넘는다.
  - ridge를 넘으면 시간이 FLOP에 비례하므로, 버려지는 12–13토큰(16 − τ)의 연산이 그대로 비용이
    된다.
  - 9B DF verify는 fla 이후에도 작은 B에서 대역폭 활용률이 48–52%에 그친다. 이는 GDN block의
    kernel launch 대기 때문이다(4.2절).
- **S가 길면 8B는 ridge에 닿지 않는다.** KV 읽기가 byte를 지배해 intensity가 KV-only 선인
  16·32/8=64에 붙는다. 대신 q=16 attention kernel이 시간을 정한다. 32k B=1에서 attention kernel이
  q=1은 7.1 ms, q=16은 27.8 ms로 3.9배다(`batch_verify_breakdown`).

### 2.3 스케줄링: static batch의 꼬리

DFlash 행은 256/τ step 만에 끝나는데 τ가 행마다 달라 끝나는 시점이 흩어진다. 반면 static batch는
마지막 행이 끝날 때까지 모든 행을 verify한다. baseline은 모든 행이 정확히 256 step이라 꼬리가 없다.

| | B=1 | B=4 | B=16 | B=32 |
| --- | --- | --- | --- | --- |
| 8B 4k: 행-step당 토큰 / τ | 0.98 | 0.77 | 0.72 | 0.66 |
| 9B 4k: 행-step당 토큰 / τ | 0.99 | 0.79 | 0.70 | 0.71 |
| 9B 8k: 행-step당 토큰 / τ | 0.98 | 0.74 | 0.57 | 0.57 |

모든 행이 살아 있는 구간만 보면 9B 4k의 speedup이 B=8까지 1.97–1.99로 유지된다. makespan은
1.51–1.97이다. 따라서 **작은~중간 B에서 makespan speedup을 깎는 주범은 꼬리다.**

### 2.4 결론: 구간별 주 병목

| 구간 | 1순위 | 2순위 | 3순위 |
| --- | --- | --- | --- |
| 작은 B, 짧은 S | τ 천장 (16토큰 중 3.4–4.2개 채택) | draft forward + LM head | 9B: GDN rollback |
| 작은 B, 긴 S | τ 하락 (8B: 3.44 → 1.94) | 8B: q=16 attention (c_verify 1.60) | draft |
| 중간 B (2–8) | static batch 꼬리 (×0.67–0.88) | 9B: rollback | draft |
| 큰 B (≥16) | **compute-bound verify** (DF verify가 BL의 2–3배) | 꼬리 (×0.57–0.72) | draft 증가 (BL step의 0.43–0.69배) |

추론 경로로 옮기면 한 decode step은 다음 순서로 흐른다.
1. **verify** (`batch.py:800`) — DF step의 58–87%(8B 79–87%, 9B 58–70%). 큰 B에서 compute-bound
   비용이 난다.
2. **commit** (`:823`) — 9B의 GDN replay가 여기서 일어난다.
3. **context-feature concat** (`:855`) — 시간은 무시할 수준.
4. **draft forward** (`:876`)
5. **LM head** (`:886`) — B=1에서 계측된 draft stage의 54–58%. 큰 B·긴 S에서는 draft attention이
   더 크다.
6. 루프 밖에서는 끝난 행을 끌고 가는 **static batch**가 꼬리 비용을 만든다.

---

## 3. 메모리 오버헤드 상세 (`batch_memory_*`)

### 3.1 성분 계수

| 성분 | 8B | 9B | B 의존 | S 의존 | 수명 | DFlash 전용 |
| --- | --- | --- | --- | --- | --- | --- |
| target weight | 15.26 GiB | 16.68 GiB | — | — | 상주 | 아니오 |
| draft weight | 1.95 GiB | 2.41 GiB | — | — | 상주 | 예 (측정상 BL에도 적재됨, I7) |
| target KV | 144 KiB/token | 32 KiB/token (full-attn 8층) | B·S | ✓ | 상주 (setup에서 할당) | 아니오 |
| GDN state | — | 49.5 MiB/행 | B | — | 상주 | 아니오 |
| **draft KV** | **20 KiB/token** (target KV의 14%) | **24 KiB/token** (target KV의 75%) | **B·S** | ✓ | 상주 (setup에서 할당) | **예** |
| prefill activation | 105 KiB/token | fallback 217 → fla 141 KiB/token | 1행 | ✓ | target prefill 중 | 아니오 |
| **tapped hidden** | **40 KiB/token** (tap 5개) | **64 KiB/token** (tap 8개) | 1행 | ✓ | target prefill ~ drafter prefill | **예** |
| context feature (concat) | tapped hidden과 같은 크기 | 같은 크기 | 1행 | ✓ | concat 순간에 hidden과 공존 (**2배**) | **예** |
| decode rollback scratch | 무시할 수준 | DF 약 87 MiB/행, BL 약 52 MiB/행 | B | — | commit 중 | DF 추가분 약 35 MiB/행 |

draft KV는 층당 4 KiB/token이다. 8B drafter는 5층 모두 full attention이다. 9B drafter는 full 1층 +
sliding 5층(window 4096)인데, **엔진이 sliding 층도 전체 길이로 저장한다**(I5). 이상적인 ring
buffer라면 `S·4 KiB + 5·4096·4 KiB`/행이다.

### 3.2 피크는 어느 phase에서 나는가

아래는 static 할당(setup 이후) 위로 각 phase의 최고점이 얼마나 올라가는지를 GiB로 나타낸 표다.

| 점 | target prefill | ctx-feature build | drafter prefill | decode verify | decode accept/rollback | 피크 phase |
| --- | --- | --- | --- | --- | --- | --- |
| 8B 4k B=1 | **0.56** | 0.31 | 0.31 | 0.01 | 0.01 | target prefill |
| 8B 4k B=32 | **0.56** | 0.31 | 0.31 | 0.20 | 0.34 | target prefill |
| 8B 32k B=1 | **4.52** | 2.50 | 2.50 | 0.01 | 0.01 | target prefill |
| 9B-fb 32k B=1 | **8.53** | 4.00 | 3.25 | 0.08 | 0.08 | target prefill |
| 9B 32k B=1 | **6.16** | 4.00 | 3.25 | 0.08 | 0.08 | target prefill |
| 9B 4k B=16 | 0.77 | 0.50 | 0.41 | 1.24 | **1.36** | decode accept/rollback |
| 9B 4k B=32 | 0.77 | 0.50 | 0.41 | 2.48 | **2.71** | decode accept/rollback |

- **target prefill이 피크가 되는 이유.** 요청을 한 행씩 prefill하고 KV는 이미 모든 행분 할당해
  두었다. 그래서 피크는 "static 할당 + 한 행의 가장 큰 transient"가 된다.
  - 가장 큰 transient는 target이 S토큰 전체를 MLP(12288차원)와 attention에 통과시키는 순간이다.
    9B는 GDN chunk 연산이 더해진다.
  - **바로 그 순간 tapped hidden이 함께 살아 있다.** DF와 BL의 target prefill 초과분 차이는
    정확히 tapped hidden 크기다(8B 32k: 4.52 − 3.27 = 1.25 GiB = 40 KiB × 32768).
- **drafter prefill이 target prefill보다 작은 이유.** drafter는 context 토큰을 fc와 층별 K/V
  projection에만 통과시키고, query·attention·MLP에는 넣지 않는다. 그 대신 ctx-feature concat 순간에
  hidden과 feature가 공존해 **hidden 크기의 2배**가 된다(8B 32k 2.50 GiB = 80 KiB × 32768). 그래도
  target activation보다는 작다.
- **9B는 작은 S·큰 B에서 decode가 피크가 된다.** commit 중 행당 약 87 MiB가 필요하다.
  - 구성 (추정): replay용으로 캡처한 (q, k, v, g, β) 24층분(행당 약 9 MiB) + 새 recurrent state
    사본(24층 × 2 MiB) + kernel scratch. BL도 state 사본(약 52 MiB/행)은 가진다.
  - 이 값이 한 행의 prefill transient(0.77 GiB)를 B≈9에서 넘어선다. 그래서 4k B≥16, 8k B=32의
    피크가 decode로 옮겨 간다.
  - 이 transient는 **fla 전후가 똑같다**(2.48 / 2.71 GiB). kernel이 아니라 엔진의 capture 구조에서
    생기는 비용이다.

### 3.3 DF−BL 피크 차이의 분해

`ΔPeak = draft KV + tapped hidden(1행)`이 8B는 전 점에서 오차 0.1 GiB 이내로 성립한다. 9B는
합이 0.1–0.25 GiB 크다. 피크 순간의 activation(unattributed)이 BL보다 그만큼 작게 잡혔다.

| 점 | ΔPeak | draft KV | tapped hidden | 합 |
| --- | --- | --- | --- | --- |
| 8B 4k B=32 | +2.91 | 2.69 | 0.16 | 2.85 |
| 8B 8k B=16 | +2.94 | 2.59 | 0.31 | 2.90 |
| 8B 32k B=4 | +3.78 | 2.52 | 1.25 | 3.77 |
| 9B 16k B=16 | +6.99 | 6.11 | 1.00 | 7.11 |
| 9B 32k B=8 | +7.81 | 6.06 | 2.00 | 8.06 |
| 9B 4k B=32 (decode 피크) | +4.33 | 3.22 | — | + rollback scratch (DF−BL) 1.09 |

- **B에 비례해 커지는 DFlash 전용 항은 draft KV뿐이다.** 8B는 +0.08 GiB(4k B=1)에서 +2.69 GiB(4k
  B=32)가 되고, 9B는 +0.76 GiB(32k B=1)에서 +6.06 GiB(32k B=8)가 된다.
- **tapped hidden은 S에만 비례하고 B와 무관하다.**
- 9B는 decode 피크 구간에서 rollback scratch가 B에 비례하는 두 번째 항이다.

### 3.4 Hidden state 주입의 메모리: 원래 가설 판정

| 가설 | 판정 | 근거 |
| --- | --- | --- |
| S가 길어지면 hidden state materialization이 커진다 | **맞음** | 8B 40, 9B 64 KiB/token. 32k에서 1.25 / 2.00 GiB |
| 그 overhead가 B와 함께 커진다 | **이 엔진에서는 틀림** | 요청 단위 순차 prefill이라 한 행분만 존재한다. 여러 요청을 묶어 prefill하는 serving에서는 동시에 prefill하는 토큰 수에 비례한다 |
| 층별 주입이 overhead를 만든다 | **형태가 다름** | hidden을 층마다 복사하지 않는다. fc가 한 번 투영하고 각 층의 K/V projection 결과가 draft KV에 저장된다. **주입의 메모리 비용은 draft KV 자체**이며 B·S에 비례한다 |
| 첫 drafter prefill에서 긴 시퀀스 KV가 overhead가 된다 | **부분적으로 맞음** | draft KV는 prefill 때 채우지만 공간은 setup에서 이미 할당됐다. 그래서 피크를 만드는 것이 아니라 **바닥을 올리고**, OOM은 setup 할당 단계에서 난다. drafter prefill의 transient(concat 2배)는 target prefill보다 작다 |
| 피크가 target prefill이므로 drafter overhead는 작다 | **틀림** | target prefill 피크 안에 tapped hidden이 들어 있고(ΔPeak의 일부), draft KV는 모든 phase의 바닥을 올린다 |

### 3.5 용량 경계와 OOM

| | 8B BL / DF max B | 9B BL / DF max B | DF만 OOM인 점 |
| --- | --- | --- | --- |
| 4k | 32 / 32 | 32 / 32 | — |
| 8k | 16 / 16 | 32 / 32 | — |
| 16k | 8 / 8 | 32 / 16 | 9B 16k B=32 |
| 32k | 4 / 4 | 16 / 8 | 9B 32k B=16 |

- 모든 OOM은 `setup: allocate caches`에서 났다. **static 할당만으로 카드를 넘은 것이다.**
  - 9B 16k B=32 DF의 static 할당: weight 19.1 + target KV 16.3 + GDN state 1.6 + draft KV 12.2 =
    49.1 GiB
  - 같은 점의 BL은 36.9 GiB라 들어간다.
- 8B는 draft KV가 target KV의 14%라 한 단계 차이를 만들지 못한다. 9B는 target KV가 작고(hybrid)
  draft KV가 75%라 **drafter가 한 단계를 잃는다.**
- **ring buffer 반사실 (추정).** 9B drafter의 sliding 5층을 window 크기로 저장하는 경우다.

  | 점 | draft KV | 추정 피크 (fallback / fla) |
  | --- | --- | --- |
  | 16k B=32 | 12.21 → 4.54 GiB | 45.7 / 44.5 GiB — 둘 다 들어감 |
  | 32k B=16 | 12.11 → 3.27 GiB | 47.8 / 45.4 GiB — **fla에서만** 들어감 |

  fla가 prefill transient를 줄였기 때문에, 두 조치를 **함께** 해야 32k×16이 열린다.

### 3.6 Iso-memory: 카드 한 장의 최고 throughput

| | 4k | 8k | 16k | 32k |
| --- | --- | --- | --- | --- |
| 8B | 0.85× | 0.93× | 1.01× | 0.77× |
| 9B-fb | 0.93× | 0.73× | 0.68× | 0.83× |
| 9B | 0.87× | 0.67× | 0.63× | 0.86× |

메모리가 허락하는 가장 큰 B에서는 baseline이 이긴다. 9B는 draft KV 때문에 16k와 32k에서 DF의 최대
B가 한 단계 낮아 격차가 더 크다. 참고로 baseline 피크에는 사용하지 않는 draft weight(2.41 GiB)가
포함되어 있다(I7). 그래서 이 비교는 오히려 DFlash에 유리한 쪽으로 기울어 있다.

### 3.7 fla가 메모리에 준 영향

- **target prefill activation만 줄었다.** 9B BL 기준 217 → 141 KiB/token이고, 32k B=1 DF 피크는
  29.44 → 27.07 GiB가 되었다.
- **static 할당, draft 쪽 성분, decode rollback scratch는 변하지 않았다.** 그래서 OOM 경계도
  그대로다.

---

## 4. fla 적용 전후 Qwen3.5-9B

### 4.1 추론 방식의 차이

| 경로 | fallback (`record_batch/`) | fla (`record_batch_fla/`) |
| --- | --- | --- |
| DF verify (q=16), GDN 24층 | `torch_chunk_gated_delta_rule`: chunk 단위 Python 루프가 작은 elementwise·bmm kernel을 다수 launch. 입력을 fp32로 올려 계산. 엔진이 chunk_size=16을 강제 | `fla.ops.gated_delta_rule.chunk`: Triton chunk kernel(내부 chunk 64, 엔진의 16은 무시됨). bf16 입력 + fp32 누적 |
| BL decode (q=1) | `torch_recurrent_gated_delta_rule`: 한 토큰도 여러 elementwise 연산 | `fused_recurrent`: 층당 fused kernel 하나 |
| DF rollback (commit) | 채택된 prefix를 **Python 루프로 replay**. 비용이 max kept에 비례 → B가 크면 max kept도 커져 29 → 47 ms | 같은 replay를 kernel 한 번으로 처리. **B와 무관하게 약 20 ms** |
| BL commit | 모든 recurrent state에 `torch.where` (fla와 무관) | 같음 (B=32에서 12.5 ms) |
| conv1d | torch (`causal_conv1d` 미설치) | torch (같음) |
| drafter | dense attention, fla와 무관 | 같음 |
| 수치 | fp32 recurrence | bf16 kernel. 최종 state 상대오차 약 0.7% |

### 4.2 결과 차이

**Speedup (makespan):**

| S | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
| --- | --- | --- | --- | --- | --- | --- |
| 4k | 1.50 → **1.97** | 1.33 → **1.75** | 1.09 → **1.57** | 1.08 → **1.51** | 1.00 → 1.06 | 0.93 → **0.87** |
| 8k | 1.49 → **1.88** | 1.23 → **1.55** | 1.06 → **1.39** | 1.03 → **1.24** | 0.82 → 0.82 | 0.73 → **0.67** |
| 16k | 1.44 → **1.88** | 1.22 → **1.64** | 1.13 → **1.45** | 1.18 → **1.33** | 0.90 → 0.89 | DF OOM |
| 32k | 1.32 → **1.68** | 1.13 → **1.54** | 1.06 → **1.40** | 1.19 → **1.29** | DF OOM | |

**Step 시간 (4k, ms):**

| | B=1 | B=8 | B=32 |
| --- | --- | --- | --- |
| DF step | 111.1 → 76.8 | 120.3 → 81.5 | 245.5 → 201.2 |
| ㄴ verify | 72.0 → 46.6 | 72.6 → 47.7 | 164.6 → 140.6 |
| ㄴ accept/rollback | 29.4 → 19.4 | 35.6 → 20.1 | 47.3 → 23.2 |
| ㄴ draft | 10.1 → 10.9 | 12.3 → 13.7 | 33.9 → 37.5 |
| BL step | 40.4 → 36.9 | 43.2 → 38.0 | 79.0 → 58.8 |
| DF step / BL step | 2.75 → **2.08** | 2.78 → **2.14** | 3.11 → **3.42** |

**Verify forward 분해** (고정 shape 프로파일, ms, `batch_verify_breakdown{,_fla}`):

| shape | wall | GDN block | GPU idle (wall − kernel) |
| --- | --- | --- | --- |
| 4k B=1 q=16 | 89.0 → 56.1 | 63.5 → 30.0 | 52.1 → 25.7 |
| 4k B=16 q=16 | 95.8 → 82.4 | 45.3 → 28.0 | 4.3 → 5.7 |
| 32k B=1 q=16 | 90.3 → 56.7 | 54.7 → 22.4 | 43.7 → 16.5 |
| 32k B=8 q=16 | 89.8 → 67.0 | 43.7 → 16.8 | 18.8 → 5.7 |
| 4k B=1 q=1 | 35.9 → 32.2 | 14.6 → 10.6 | 7.5 → 5.2 |

**변하지 않은 것:** τ(점마다 ±0.13 이내), DF==BL 출력 일치(12–25/32 → 15–24/32로 bf16 수치 차이
수준), OOM 경계, static 할당, decode scratch.

### 4.3 해석과 갱신해야 할 기존 결론

- **fallback은 baseline을 kernel-bound로 묶어 두고 있었다.** fla로 BL 9B verify가 copy BW의
  71–73%까지 올라갔다. 큰 B에서는 BL이 DF보다 더 많이 빨라졌고(B=32에서 BL −26%, DF −18%), 그래서
  큰 B의 speedup이 내려갔다.
- **작은 B에서는 DF의 고정 비용이 줄어** 비율이 2.75에서 2.08로 내려갔다.
- 앞선 분석에서 "9B speedup은 fallback 고정 비용이 지배해 B에 둔감하다"고 했는데, **이는 fallback이
  만든 착시였다.** fla 이후 9B는 8B와 같은 모양을 보인다. B≈8까지 비율이 평탄하다가 B=16에서
  compute-bound로 전이한다.
- **남은 DF 전용 비용 1위는 GDN rollback이다.** 약 20 ms로 DF step의 25%, BL step의 0.48배다. verify
  안에도 GPU idle이 26 ms 남아 있다(작은 B, q=16).
- PROFILING2.md의 9B prefill activation 계수(약 210 KiB/token)는 fla 환경에서 141 KiB/token으로
  바꿔 읽어야 한다.

---

## 5. 경향

### 5.1 S가 길어질 때 (B=1 기준)

| | 4k | 8k | 16k | 32k |
| --- | --- | --- | --- | --- |
| 8B speedup | 2.49 | 2.12 | 1.68 | 1.05 |
| 8B τ (전체 / EOS 전) | 3.44 / 3.01 | 3.17 / 2.97 | 2.67 / 2.27 | 1.94 / 1.48 |
| 8B c_verify | 1.09 | 1.19 | 1.31 | 1.60 |
| 9B speedup | 1.97 | 1.88 | 1.88 | 1.68 |
| 9B τ (전체 / EOS 전) | 4.24 / 3.76 | 4.04 / 3.64 | 4.07 / 4.03 | 3.69 / 3.08 |
| 9B c_verify | 1.33 | 1.34 | 1.38 | 1.40 |

- **8B(attention 36층)는 두 요인이 겹쳐 speedup이 크게 줄어든다.**
  - τ가 떨어진다.
  - 16토큰 verify의 attention이 긴 KV를 읽으면서 c_verify가 오른다. 32k에서 attention kernel이
    q=1 대비 3.9배다.
  - draft 쪽도 비슷하다. 8B drafter는 5층 모두 full attention이라, 32k B=4에서 draft attention이
    계측된 stage의 81%다.
- **9B(attention 8층)는 c_verify가 거의 평탄하다.** τ가 완만하게 떨어지는 것이 주된 요인이다.
- **메모리.** B·S에 비례하는 항(target KV, draft KV)과 한 행의 S 항(prefill activation, tapped
  hidden 2배)이 함께 커진다.
  - 8B의 max B는 32 → 16 → 8 → 4가 된다.
  - 9B는 16k부터 draft KV 때문에 DF가 한 단계를 잃는다.
- **주의.** S마다 prompt 구성이 다르다. 16k는 32행 중 repobench-p(코드)가 11개다. 32k는 여러 문서를
  이어 붙인 합성 prompt 후보가 46개 들어간 pool(다른 S는 0)에서 뽑혀 multi-hop QA 위주이고, 32행
  모두 자연 EOS에 도달했다. τ-S 관계에는 task 효과가 섞여 있다(M5).

### 5.2 B가 커질 때 (4k 기준)

| | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
| --- | --- | --- | --- | --- | --- | --- |
| 8B speedup (makespan / full) | 2.49 / 2.49 | 2.13 / 2.39 | 1.80 / 2.21 | 1.62 / 2.07 | 1.05 / 1.32 | 0.85 / 1.11 |
| 9B speedup (makespan / full) | 1.97 / 1.97 | 1.75 / 1.98 | 1.57 / 1.99 | 1.51 / 1.98 | 1.06 / 1.51 | 0.87 / 1.21 |
| 8B DF step / BL step | 1.36 | 1.40 | 1.48 | 1.57 | 2.35 | 2.68 |
| 9B DF step / BL step | 2.08 | 2.16 | 2.15 | 2.14 | 2.79 | 3.42 |
| 8B ΔPeak (GiB) | +0.24 | +0.33 | +0.50 | +0.85 | +1.53 | +2.91 |
| 9B ΔPeak (GiB) | +0.32 | +0.42 | +0.62 | +1.03 | +2.17 | +4.33 |

- **τ는 B와 무관하다.** 배치 크기가 채택률에 영향을 주지 않는다.
- **B≤8:** 비율이 평탄하고 speedup 감소는 **꼬리** 때문이다.
- **B≥16:** 비율이 뛰고 **compute-bound verify**와 draft 증가가 더해진다.
- **메모리:** ΔPeak는 draft KV 때문에 B에 선형으로 늘어난다. 9B는 decode scratch도 B에 비례한다.

---

## 6. 병목 분류

### 6.1 연구 관점: 블록 diffusion drafter + hidden state 투영 계열 자체의 병목

**R1. 고정 폭 블록 검증과 채택률의 불일치**
- **현상.** 매 step B×16 위치를 검증하지만 채택은 τ≈3.4–4.2다. 검증 토큰의 74–79%가 버려진다.
  memory-bound 영역(B≤8)에서는 거의 공짜지만, compute-bound 영역(B≥16)에서는 버려지는 토큰의
  FLOP이 그대로 시간이 된다(2.2절: 8B 비율 1.57 → 2.35, 9B 2.14 → 2.79).
- **이 계열에 고유한 이유.** diffusion 계열 drafter는 블록 전체를 한 번에 병렬로 내놓고, 자연스러운
  검증도 블록 전체다. 자기회귀 drafter처럼 "필요한 만큼만 초안"을 만드는 구조가 아니다.
- **연구 방향.**
  - B와 bound 영역에 따라 검증 폭을 적응적으로 조절한다.
  - drafter의 위치별 신뢰도로 블록을 행마다 잘라낸다.
  - 배치 전체의 토큰 예산을 신뢰도가 높은 행에 몰아준다.
  - 목표는 "B·q ≤ ridge"를 유지하는 스케줄링이다.

**R2. Drafter 비용도 B·q에 비례한다 (특히 전체 어휘 LM head)**
- **현상.**
  - draft forward + logits가 BL step 대비 8B 0.27 → 0.43, 9B 0.30 → 0.64(B=1 → 32)로 커진다.
  - 계측된 draft stage 중 LM head가 B=1에서 54–58%다(9B 어휘 248k × 15위치).
  - 큰 B에서 draft forward도 GEMM 최대 성능의 51–56%에 도달한다(compute 영역).
- **이 계열에 고유한 이유.** 병렬 drafter는 매 step 모든 블록 위치의 logits가 필요하다. 채택될
  가능성이 낮은 뒤쪽 위치도 똑같이 비용을 낸다.
- **연구 방향.** 어휘 shortlist나 저랭크 head, 위치별로 head를 생략하는 early exit, B 영역에 맞춘
  drafter 폭·깊이 설계.

**R3. Context를 K/V로 주입하는 구조가 drafter 전용 full-context KV를 요구한다**
- **현상.** draft KV는 B·S에 비례하는 static 항이다(8B target KV의 14%, 9B 75%).
  - 9B에서 DF만 OOM이 나는 두 점을 결정한다.
  - drafter가 full attention이면 긴 S에서 draft attention 비용도 커진다(8B 32k B=4: stage의 81%).
- **이 계열에 고유한 이유.** DFlash는 target hidden을 모든 draft 층의 K/V로 주입해 전체 문맥을
  조건으로 삼는다. hybrid·linear·SWA target이 없앤 O(S) KV를 **drafter가 다시 도입한다.** target
  아키텍처가 효율적일수록 상대 overhead가 커진다.
- **연구 방향.**
  - drafter의 문맥 창·sink 설계. 9B drafter는 이미 sliding 5층이다. full 1층까지 창으로 바꿔도 τ가
    유지되는지 검증할 가치가 있다.
  - target KV를 재사용하거나 투영해 공유한다.
  - 층 간 KV 공유, draft KV 양자화.

**R4. Multi-layer hidden tap의 prefill materialization**
- **현상.**
  - prompt 전체의 tap(tap 수 × H × S)이 target prefill activation과 동시에 존재한다. target
    prefill 피크를 8B 40, 9B 64 KiB/token만큼 올린다.
  - concat 순간에는 그 2배가 된다.
  - 이 엔진은 순차 prefill이라 한 행분이지만, chunked·batched prefill serving에서는 동시에 prefill하는
    토큰 수에 비례한다.
- **이 계열에 고유한 이유.** EAGLE-3나 DFlash처럼 여러 층의 hidden을 조건으로 쓰는 drafter는 prompt
  전체의 multi-layer 표현을 한 번은 만들어야 한다.
- **연구 방향.** tap 수와 층 선택의 최소화, 저랭크 tap 투영, prefill 구간에서 조건을 압축하는 방법.
  (fc(concat) = Σ Wᵢhᵢ를 이용해 층마다 누적하는 것은 수학적으로 정확한 재작성이므로 구현 항목 I6에
  둔다.)

**R5. Stateful target 층의 rollback 비용**
- **현상.** 9B에서 GDN rollback은 fla 이후에도 약 20 ms 고정이다(DF step의 25%, 작은 B에서 BL step
  대비 0.48배). decode 중 행당 약 35 MiB의 scratch도 추가로 필요하고, 작은 S·큰 B에서 이것이 피크가
  된다.
- **이 계열에 고유한 이유.**
  - attention은 길이만 자르면 되지만, recurrent state는 채택 위치의 state를 다시 만들거나 위치마다
    저장해야 한다.
  - 블록 폭 q가 크면 replay 구간과 캡처 크기도 커진다.
  - 이것은 drafter 종류가 아니라 **"큰 블록 검증 × stateful target"의 조합**에서 생긴다.
- **연구 방향.**
  - 검증 kernel이 채택 가능한 위치의 중간 state를 함께 내보내게 한다(replay 제거, 메모리 O(q × state)).
  - 채택 길이 예측으로 캡처 범위를 줄인다.
  - recurrent 층에 맞는 검증 알고리즘을 설계한다.

**R6. 긴 문맥에서의 채택률 하락**
- **현상.** EOS 이전 τ 기준으로 8B는 3.01 → 1.48(4k → 32k), 9B는 3.76 → 3.08이다.
- **이 계열에 고유한 이유.** drafter는 target보다 훨씬 작은 모델로 긴 문맥 조건을 따라가야 한다.
  학습 문맥 길이와 위치 처리에 민감할 가능성이 크다. 단, prompt 구성이 S마다 달라 혼재 요인이 있다(M5).
- **연구 방향.** 긴 문맥 drafter 학습, 문맥 압축 조건. 먼저 task 구성을 통제한 재측정이 필요하다.

**R7. 행별 채택률 편차가 만드는 배치 꼬리**
- **현상.** 끝난 행까지 verify하면서 행-step당 토큰이 τ의 57–72%로 줄어든다. 모든 행이 살아 있는
  구간의 speedup(9B 4k B≤8에서 1.97–1.99)과 makespan(1.51–1.97)의 차이가 여기서 온다.
- **이 계열에 고유한 이유.** speculative decoding은 행마다 진행 속도가 확률적으로 다르다. 블록이 클수록
  채택 분산이 커져 행들이 더 흩어진다.
- **연구 방향.** 채택률 분산을 고려한 batch 스케줄링, 행별 적응 블록. (continuous batching 자체는
  구현 항목 I11)

### 6.2 구현 관점: 코드·kernel·측정에서 다뤄야 할 것

| # | 항목 | 영향 (측정값) | 상태 / 조치 |
| --- | --- | --- | --- |
| I1 | GDN torch fallback | 9B DF verify 72 ms, rollback 29–47 ms, BL은 kernel-bound(copy BW 49–63%) | **해결** (fla). verify 47 ms, rollback 약 20 ms, BL 71–73% |
| I2 | fla 경로에 남은 kernel launch 대기 | 4k B=1 q=16 verify 56 ms 중 GPU idle 26 ms | 미해결. CUDA graph나 torch.compile로 decode step 캡처 |
| I3 | Rollback replay 구현 | 24층을 각각 replay하고 mask 연산 추가, 약 20 ms 고정 | 미해결. 전 행이 블록 전체를 채택하면 생략, 층 묶기, 중간 state 출력 kernel 사용 (R5와 연결) |
| I4 | **9B baseline의 `torch.where` commit** | BL step의 5–21%(B=32에서 12.5 ms). **speedup이 DFlash 쪽으로 과대평가됨** | 미해결. 없애면 (추정) 4k B=1 1.97 → 1.88, B=8 1.51 → 1.35, B=16 1.06 → 0.89, B=32 0.87 → 0.68, 8k B=32 0.67 → 0.54 |
| I5 | 9B drafter sliding 층의 전체 길이 저장 | draft KV 24 KiB/token(이상적이면 약 4 KiB/token + 창). DF만 OOM인 두 점 | 미해결. ring buffer (3.5절: fla와 함께면 두 점 모두 들어감, 추정) |
| I6 | context feature concat의 2배 materialization | ctx-build 순간 hidden의 2배(8B 32k 2.50 GiB) | 미해결. 층별 fc 누적(Σ Wᵢhᵢ)으로 S·H 버퍼 하나로 줄임 |
| I7 | baseline 측정에 draft weight 적재 | BL 피크 +1.95 / 2.41 GiB. iso-memory·피크 비교가 DFlash에 유리하게 기움 | 측정 편향. 이 격자에서 max B는 불변. 순수 BL 프로세스로 재측정하면 확정 |
| I8 | Static KV 사전 할당 | OOM이 전부 setup 할당에서 발생. 용량이 실제 토큰 수가 아니라 할당 정책으로 정해짐 | 설계 선택. paged KV나 필요 시 할당 |
| I9 | 요청 단위 순차 prefill | hidden tap이 B와 무관한 것은 이 설계 덕분. prefill이 wall time의 절반 이상 | serving 형태(batched·chunked prefill)로 바꾸면 R4가 B에 따라 드러남 |
| I10 | q=16 attention kernel | 8B 32k B=1에서 attention kernel이 q=1 대비 3.9배(7.1 → 27.8 ms). GQA 그룹 간 KV 공유 없이 읽는 것으로 추정 | 미해결. verify 모양(짧은 query, 긴 KV)에 맞는 kernel |
| I11 | Static batching | 꼬리 비용 ×0.57–0.72 | 미해결. continuous batching |

**측정 관련 문제와 갱신 사항**

| # | 항목 | 내용 | 상태 |
| --- | --- | --- | --- |
| M1 | fla-core 단독 설치 시 조용한 fallback | transformers의 getattr 체인이 `fla.ops`를 찾지 못해 torch 함수가 그대로 남음. 경고 없음 | **해결.** 두 패키지 설치, 레코드에 `gdn_kernels` 기록, 그림 각주 자동 표기 |
| M2 | 엔진이 넘기는 `chunk_size=16`이 fla에서 무시됨 | fla는 내부 64 사용. 결과에 무해 | 기록만 |
| M3 | GC 참조 순환 누수, draft mask cache 충돌 | 9B 피크 과대, τ 과소 | **해결.** `stale_*`로 격리하고 재측정 (plan 부록 B) |
| M4 | speedup 분해에 전체 τ 사용 | EOS 이후 τ가 높고 끝난 행도 verify되므로 과대평가 | **해결.** 행-step당 유효 토큰으로 교체 (항등식이 전 점에서 성립) |
| M5 | S별 prompt 구성 차이 | 32k는 합성 prompt pool에서 multi-hop QA 위주, 16k는 32행 중 코드 11개. τ-S 관계에 task 효과가 섞임 | 미해결. S 간 동일 task 구성으로 재측정 |
| M6 | EOS 억제 | EOS 이후 τ가 EOS 이전보다 0.1–0.9 높아 전체 τ가 낙관적. 반대로 baseline에는 꼬리가 없어 makespan은 DFlash에 비관적 | 두 편향의 방향이 반대. natural-EOS 모드로 보조 측정 필요 |
| M7 | draft stage 계측 범위 | stage event가 draft 호출 시간의 36–70%만 덮음(MLP와 noise 토큰 projection 미계측) | 부분 계측. R2의 LM head 비중은 "계측된 stage 중"으로만 읽을 것 |
| M8 | `cpu_and_sync`가 음수 | GPU event 구간이 겹쳐 최대 0.3 ms 과다 집계 | 무시 가능한 수준 |
| M9 | analytic byte/FLOP | 이상적 kernel 기준의 하한. 지붕까지의 거리는 kernel이나 launch 비효율을 뜻함 | 해석 규칙 (README) |
| M10 | bf16 출력 일치율 약 50% | 배치 모양에 따른 수치 차이. BL 자신도 B=1 대비 같은 수준. 무손실성은 CPU fp32 테스트와 τ 안정성으로 확인 | 기록만 |
| M11 | 9B 기존 결론 갱신 | "B에 둔감", "고정 비용 지배", "rollback이 B와 함께 증가", "prefill act 약 210 KiB/token", iso-memory 값 | **갱신됨** (4.3절). 두 스윕의 점은 섞지 말 것 |
| M12 | verify breakdown은 별도 profiler pass | throughput 표와 섞지 않음. fla용은 `record_batch_fla/verify_breakdown_qwen3.5-9b.json` | 기록만 |

---

## 7. 다음 실험 제안

1. **I4 baseline commit 제거.** 가장 싸고, 9B 큰 B 결론을 바로 바꾼다.
2. **I5 ring buffer.** 9B의 두 OOM 점을 열고, R3의 "drafter 문맥 창" 연구의 기준선이 된다.
3. **I3/R5 rollback.** 전 행이 블록 전체를 채택하면 replay를 생략하고, 층별 호출을 묶는다. 작은 B의
   9B 비율 2.08을 얼마나 낮추는지 본다.
4. **R1 검증 폭 스윕.** 블록 {4, 8, 16} × B. B마다 최적 폭이 달라지는지(compute-bound 영역에서 작은
   폭이 이기는지) 확인한다.
5. **M5/M6 통제.** S 간 task 구성을 같게 하고, natural-EOS 모드로 보조 측정한다.
6. **continuous batching 또는 행별 조기 종료(I11/R7).** 꼬리를 제거했을 때의 speedup을 측정한다.

---

## 부록: 그림과 재현

| 그림 | 이 문서의 절 |
| --- | --- |
| `batch_bound_<model>{,_fla}` | 2.2 |
| `batch_step_<model>{,_fla}` | 2.1, 2.3, 4.2 |
| `batch_throughput_<model>{,_fla}` | 2.3, 3.6, 5 |
| `batch_memory_<model>{,_fla}` | 3 |
| `batch_verify_breakdown{,_fla}` | 2.2, 4.2 |

```bash
python -m dflash.batch_report record_batch            # 표 (fallback, 두 모델)
python -m dflash.batch_report record_batch_fla        # 표 (fla, 9B)
python visualization_batch/plot_bound.py
python visualization_batch/plot_bottlenecks.py
python visualization_batch/plot_bound.py       --records record_batch_fla --suffix _fla
python visualization_batch/plot_bottlenecks.py --records record_batch_fla --suffix _fla
```
