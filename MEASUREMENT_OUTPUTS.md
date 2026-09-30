# 측정 산출물과 측정 방식 (멀티 GPU 기준, protocol v2)

- 대상: `python -m dflash.arch` (measurement protocol v2, `MEASUREMENT_AUDIT.md` F절)의 모든 출력 파일
- 작성일: 2026-09-30
- 전제: 기본 실행은 `--num-gpus 2`. target은 layer 단위로 2장에 나뉘고(pipeline식 샤딩, tensor parallel 아님), B=1.

이 문서는 각 출력 값이 **어떤 API로, 어느 장치에서, 어느 구간을** 재는지와, 여러 GPU에 나뉘어 있다는 사실이 그 값을 어떻게 바꾸는지를 정리한다. 값끼리 비교해도 되는지는 4절의 규칙을 따른다.

---

## 1. 멀티 GPU 구성: 무엇이 어디에 있는가

### 1.1 장치 선택

| 단계 | 방법 | 기록 위치 |
|---|---|---|
| 카드 고르기 | CUDA 초기화 **전에** NVML로 카드를 읽고, 정상(utilization 조회 성공)이면서 유휴(사용 메모리 1 GiB 이하, compute process 없음)인 카드 N장을 고른다 | `gpu_selection.all_cards_at_selection`, `chosen` |
| 고정 | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<UUID들>`을 설정한다. 이후 프로세스 안에서는 `cuda:0..N-1`만 보인다 | `gpu_selection.cuda_visible_devices` |
| 확인 | CUDA가 올라온 뒤 열 수 있는 장치 수가 N인지 확인한다 | `gpu_selection.devices` |

프로세스 안의 모든 `range(torch.cuda.device_count())` 루프(피크 리셋, 피크 읽기, PeakTracker)는 이 N장만 돈다. 그래서 장치별 배열의 길이는 N이고, `cuda:i`는 선택된 i번째 카드를 가리킨다. nvidia-smi 번호가 아니다. 카드의 실제 정체는 UUID로만 식별한다.

### 1.2 배치 (`placement.py`, `loader.py`)

| 구성 요소 | 장치 | 비고 |
|---|---|---|
| decoder layer | 연속 블록으로 균등 분할, 나머지는 뒤쪽 장치에 | 8B 18/18, 9B 16/16, 35B-A3B 20/20 |
| embed_tokens, rotary, (35B) vision tower | 첫 장치 `cuda:0` | vision tower는 텍스트 프롬프트에서 실행되지 않지만 가중치는 `cuda:0`에 상주 |
| final norm, **lm_head** | 마지막 장치 `cuda:1` | |
| **drafter 전체** | `cuda:0` | 분할하지 않음 |
| DFlash 상태(`output_ids`, context feature, draft KV) | `cuda:0` | |
| 선택된 target hidden state | 각 층의 장치에서 탭한 뒤 `cuda:0`으로 복사 | `HiddenStateTap(device=cuda:0)` |

실제 배치는 `load.target.device_map_applied`, `placement`, `placement_honoured`, `param_bytes_per_device`에 남는다.

### 1.3 한 스텝의 데이터 흐름과 장치 경계

```
AR 1 step     : cuda:0 [embed, L0..L(k-1)] -> P2P -> cuda:1 [Lk..L(n-1), norm, lm_head] -> P2P -> cuda:0 (logits)
DFlash 1 step : cuda:0 [drafter]  -> P2P -> cuda:1 [lm_head: draft logits] -> P2P -> cuda:0
                cuda:0 [verify L0..] -> P2P -> cuda:1 [.. lm_head] -> P2P -> cuda:0 ; .item() (host sync)
```

B=1 layer 샤딩에서는 한 순간에 대략 한 장치만 일한다. 다른 장치가 노는 시간은 구조적인 것이며, DFlash의 오버헤드가 아니다.

### 1.4 시간 측정의 공통 근거

accelerate는 최상위 모듈에 `AlignDevicesHook(io_same_device=True)`를 건다(`accelerate/big_modeling.py:211`). 그래서 **모든 target forward의 출력은 입력 장치(`cuda:0`)로 되돌아온다.** PyTorch의 장치 간 복사는 양쪽 스트림을 서로 기다리게 한다. 따라서 `cuda:0` 스트림의 어떤 지점이 완료되었다면, 그 지점까지 `cuda:1`에서 실행된 작업도 완료된 것이다. 아래의 모든 시간 측정은 이 성질에 기대어 **`cuda:0` 하나만** 동기화하거나 `cuda:0`에 CUDA event를 기록한다.

결과적으로 모든 구간 시간은 **벽시계(wall) 구간**이다. 두 장치의 작업 시간을 더한 값이 아니라, 두 장치가 순서대로 일한 시간과 장치 경계 복사, 대기를 모두 포함한다.

---

## 2. 출력 파일

```
record_arch_main/
  manifest.json                                       # manifest 서브커맨드
  plan.json                                           # plan 서브커맨드
  <model>/
    sweep_<gdn backend>_<N>gpu_<UTC stamp>.json       # sweep: 원자료, 이것이 정본
    csv/<위 레코드 이름>/*.csv                          # sweep이나 export가 만드는 파생 표
    traces/<UTC stamp>/<condition key>__{ar,dflash}.json.gz   # trace 패스의 원본 Kineto 트레이스
    verify_width_<N>gpu_<UTC stamp>.json              # verify-width
    taxonomy.json, rollback.json                      # taxonomy, rollback(실행마다 덮어씀)
    csv/*.csv  (디렉터리 바로 아래)                     # v1 시절 파일. 덮어써졌을 수 있으니 쓰지 말 것
queue/arch_seq_*.log                                   # 실행 로그(사람이 읽는 요약)
```

- 레코드 파일명의 `<gdn backend>`는 실제로 적용된 GDN 커널 조합(`causal_conv1d+fla` 또는 `torch_reference`)이다. `<N>gpu`는 고정한 장치 수다.
- v1 레코드(파일명에 `_Ngpu`가 없는 것)는 3-GPU 배치였고, AR을 drafter가 올라간 상태에서 쟀다. 이 레코드들의 CSV는 `export`로 `csv/<레코드 이름>/`에 다시 만들어 두었다.

### 2.1 sweep 레코드(JSON) 구조

| 최상위 키 | 내용 |
|---|---|
| `manifest` | git 커밋과 dirty 여부, 패키지 버전, GPU 인벤토리(UUID, PCI, peer access), 환경 변수 |
| `gpu_selection` | 1.1절의 선택 과정 |
| `backend`, `gdn_backend` | GDN과 conv 커널 해석 결과. `homogeneous`가 False이면 섞인 백엔드 |
| `load` | target 로드 보고(배치, 장치별 가중치 바이트), drafter 보고, `drafter_loaded_after_ar: true` |
| `target_taxonomy` | 층별 mixer(full_attention/gdn)와 FFN(dense/moe) 분류 요약 |
| `limits` | RoPE 범위, drafter 학습 컨텍스트 |
| `prompt_set` | 프롬프트 모드, 과제, S당 개수 K, seed, `same_source_across_lengths` |
| `measurement_protocol` | 이 레코드가 따른 정의(버전 2) |
| `trace_dir` | 트레이스 파일 위치 |
| `conditions[]` | 조건마다 하나: 아래 표 |

조건 항목 `conditions[i]`:

| 키 | 내용 |
|---|---|
| `key` | `<model>/S<S>/b<block>/o<out>/<policy>/p<prompt>` |
| `prompt` | 과제, source_index, 합성(composed) 여부, 토큰 해시 |
| `ar.native`, `ar.recording` | AR 기준선(PassResult: `samples`는 반복별 행, `median`은 반복 중앙값과 `__min/__max/__stdev`) |
| `ar_policy_order` | 이번 조건에서 두 정책을 실행한 순서 |
| `ar_trace`, `ar_moe` | AR의 trace, moe 결과 |
| `passes.perf/memory/trace/moe/audit/exact` | DFlash 패스별 결과 |
| `speedup.<policy>` | perf 패스와 AR 정책별 비교 |
| `pass_comparison` | perf 대 memory: 계측 비용과 acceptance 동일성 |
| `trace_perturbation` | profiler를 켠 스텝당 시간 ÷ perf 스텝당 시간(`dflash`, `ar` 각각) |
| `lossless` | audit/exact 패스를 돌린 경우만 |

---

## 3. 값별 측정 방식

표기: **[wall]** 벽시계 구간, **[dev]** 장치 위 실행 시간, **[alloc]** PyTorch 할당자 카운터, **[host]** 호스트에서 계산.

### 3.1 실행 순서와 격리(모든 패스 공통)

1. **A. AR 시간 측정:** target만 로드하고, 모든 조건의 AR native와 recording을 돈다.
2. **B. DFlash 시간 측정:** drafter를 `cuda:0`에 로드하고, 모든 조건의 perf, memory, (audit, exact) 패스를 돈다.
3. **C. 진단:** moe(AR, DFlash) 다음에 trace(AR, DFlash). **torch.profiler를 한 번이라도 실행한 프로세스는 이후의 모든 kernel launch가 느려지기** 때문에, profiler를 쓰는 패스는 반드시 마지막이다. 실측한 증가폭은 Qwen3-8B 단일 GPU AR TPOT 29.4 → 33.2 ms(+13%), 35B-A3B 2 GPU 132 → 164 ms(+24%)이며, 프로세스가 끝날 때까지 유지된다. 모든 시간 측정 행에는 `profiler_used_before`가 기록되고, 이 값은 False여야 한다.
4. 각 패스를 시작할 때 한 번 `gc.collect()`, `empty_cache()`, 장치별 `reset_peak_memory_stats`를 한다. 반복마다 하지는 않는다. 반복마다 캐시를 비우면 timed 구간 안으로 cudaMalloc이 들어오기 때문이다. 패스마다 warmup이 1회 있어서 풀을 다시 데운다.
5. 반복마다 장치별 peak 카운터만 리셋한다.

### 3.2 지연 시간 (perf 패스, AR)

| 필드 | 경로 | 측정 방식 | 멀티 GPU에서의 의미 |
|---|---|---|---|
| `time_to_first_token_s` | 둘 다 | [wall] `cuda:0` synchronize 후 `perf_counter`로 prefill 구간을 잰다 | prefill은 두 장치를 순서대로 지나고, 첫 토큰의 logits가 `cuda:0`으로 돌아올 때까지를 포함한다. DFlash는 여기에 context feature 구축과 `cuda:0`으로의 hidden 복사가 더해진다 |
| `decode_latency_s` | 둘 다 | [wall] decode 시작 전과 끝에 `cuda:0` synchronize | 두 장치의 순차 실행, P2P 복사, 호스트 공백을 모두 포함한다 |
| `time_per_output_token_s` | 둘 다 | [host] `decode_latency / (n−1)`. 첫 토큰은 prefill의 것 | 분모가 같으므로 두 경로를 직접 비교할 수 있다 |
| `time_per_output_token_over_n_s` | DFlash | `dflash_generate`의 원래 값(n으로 나눔) | 이전 기록과 비교할 때만 쓴다 |
| `total_latency_s` | 둘 다 | TTFT + decode | |
| `num_output_tokens`, `output_token_ids` | 둘 다 | greedy 출력 | 장치와 무관하다 |

AR decode 루프 안에는 호스트 동기화가 없다(`ignore_eos` 조건). 토큰은 `cuda:0`의 버퍼에 쓰이고 루프가 끝난 뒤 한 번만 읽힌다. DFlash는 verify마다 acceptance 길이를 `.item()`으로 읽으므로 **스텝당 호스트 동기화가 1회** 있다. 이 동기화는 두 장치의 파이프라인을 매 스텝 비운다. 이것은 DFlash 구현의 속성이지 측정 오류가 아니다.

### 3.3 DFlash 구간 타이머 (perf 패스)

모든 구간 타이머는 `_GpuTimer`다. 구간의 시작과 끝에 **`cuda:0` 현재 스트림에 CUDA event를 기록**하고, 512쌍마다 한 번 동기화해 `elapsed_time`을 합산한다. 동기화를 추가하지는 않는다.

| 필드 | 감싸는 구간 | 멀티 GPU에서 실제로 포함되는 것 |
|---|---|---|
| `target_verify_total_s` / `_mean_s` | verify target forward 1회(블록 전체) | `cuda:0` 층 → P2P → `cuda:1` 층과 lm_head → logits가 `cuda:0`으로 복사될 때까지. `cuda:1`의 작업은 끝의 복사를 통해 포함된다. mean = total / `num_verify_steps` |
| `draft_forward_total_s` | drafter forward(첫 호출과 이후 호출) | drafter는 `cuda:0`에만 있으므로 단일 장치 시간이다 |
| `draft_forward_first_s` | 첫 drafter 호출(프롬프트 길이 context를 draft KV로 투영) | 한 번 지불하는 비용이다 |
| `draft_forward_steady_mean_s` | 두 번째 이후 호출의 호출당 평균 | |
| `draft_logits_total_s` | draft hidden → **target lm_head(`cuda:1`)** → argmax | `cuda:0`→`cuda:1` hidden 복사, 15×vocab GEMM, 결과를 `cuda:0`으로 되돌리는 복사. v1에서는 어느 타이머에도 없었다 |
| `context_feature_decode_total_s` | verify 후 선택된 hidden을 이어 붙여 다음 context feature를 만드는 구간 | hidden은 이미 `cuda:0`에 탭되어 있다 |
| `context_feature_prefill_s` | prefill의 feature 구축과 concat | decode 지표와 섞지 않는다 |
| `attributed_decode_s` | 위 네 decode 항목의 합 | |
| `unattributed_decode_s`, `_per_step_s` | `decode_latency − attributed` | noise embedding, crop, `.item()` 동기화 대기, accelerate 훅의 호스트 시간, phase 경계 호출. 분배하지 않고 그대로 보고한다 |

주의할 점:

- 타이머 값은 [wall]이다. 여러 장치의 [dev] 시간을 더한 값이 아니다. verify 한 번에서 장치별로 얼마를 썼는지는 3.6절 trace 패스로 본다.
- event가 `cuda:0`에 기록되므로, `cuda:0`이 앞선 작업 때문에 늦게 시작하면 그 대기가 다음 구간에 들어간다. 스텝 경계에 `.item()` 동기화가 있어서 누적되지는 않는다.

### 3.4 메모리

모든 [alloc] 값은 **프로세스의 PyTorch 캐싱 할당자** 기준이다. 장치마다 따로 집계된다. CUDA context(장치당 수백 MiB)와 할당자를 거치지 않는 메모리는 포함되지 않는다. OOM 판정에 쓰이는 실제 여유 공간(`mem_get_info`)과는 이 차이만큼 다르다.

#### 3.4.1 perf 패스와 AR: 실행 후 장치별 최댓값

| 필드 | 방식 | 의미 |
|---|---|---|
| `peak_memory_device_maxima_bytes` [N] | 반복 전에 리셋, 반복 후 장치마다 `max_memory_allocated(i)` | 장치 i가 실행 중 한 번이라도 도달한 최대 할당량 |
| `peak_memory_sum_device_maxima_bytes` | 위 배열의 합 | **동시에 존재한 적 없는 최댓값들의 합이다.** 샤딩된 실행에서는 실제 동시 피크보다 크다(과대). 대신 AR과 DFlash가 같은 정의를 쓰므로 둘 사이의 차이(`memory_overhead_vs_*`)는 유효하다 |
| `peak_reserved_device_maxima_bytes` [N], `_sum_` | 같은 방식으로 `max_memory_reserved(i)` | 할당자가 드라이버에서 확보한 양. OOM을 결정하는 값에 가깝다 |

AR은 drafter를 로드하기 전에 실행되므로, AR의 장치별 값에는 drafter 가중치가 없다. DFlash의 `cuda:0` 값에는 drafter 가중치, draft KV, context feature가 포함된다.

#### 3.4.2 memory 패스: 구간 분해(PeakTracker)

memory 패스만 `memory_tracking=True, watch_layers=True`로 실행된다.

- 모든 target decoder layer에 forward pre-hook이 걸리고, phase 경계(예: `decode: target verify`)에서도 구간을 끊는다.
- 구간이 끝날 때마다 장치별 `allocated_bytes.all.peak`를 원시 바인딩으로 읽고, 그 합을 구간 피크로 삼은 뒤 장치별 peak를 리셋한다.
- **근거:** layer 샤딩에서는 한 층이 한 장치에서만 실행된다. 그래서 한 구간 안에서 장치별 최댓값의 합은 거의 동시 값이 된다. 모든 구간 중 최댓값이 동시 피크다. 단일 장치에서는 기존 정의와 정확히 같다.

| 필드 | 의미 |
|---|---|
| `peak_memory_simultaneous_bytes` | 가장 큰 구간 합 = 동시 피크 |
| `peak_memory_per_device_at_aggregate_peak_bytes` [N] | **그 구간에서의** 장치별 값. 장치별 피크가 아니다 |
| `peak_memory_device_maxima_bytes` [N] | 구간 값들의 장치별 최댓값. 3.4.1과 같은 정의지만, 계측 훅이 켜진 실행에서 잰 값이다 |
| `peak_site` | 피크가 난 구간의 이름(예: `target layer 23`), decode 토큰 위치, draft 스텝 |
| `phase_memory` | phase별 진입/종료 할당량, 가장 큰 구간 피크, 그 시점의 구성 요소 분해 |
| `peak_reserved_device_maxima_bytes` | **None.** 구간마다 peak 카운터가 리셋되므로 실행 후 reserved 피크는 마지막 구간의 값일 뿐이다. reserved 값은 perf 패스에서 읽는다 |

구성 요소 분해(`phase_memory.*.components`, `ComponentProbe`):

- target 캐시와 draft 캐시를 순회하며 각 텐서의 **storage 바이트**를 장치별로 합산한다.
- 같은 storage는 한 번만 센다(`(device, data_ptr)` 키).
- 분류: 어텐션 KV, GDN recurrent, GDN conv working/recording, draft KV, 선택된 hidden, context feature, drafter 가중치.
- 장치별 합계는 `components.per_device`에 있다. 예를 들어 target KV는 층이 있는 장치에 나뉘고, draft KV와 context feature는 `cuda:0`에만 있다.

memory 패스의 지연 시간은 결과표에 쓰지 않는다. `pass_comparison.tpot_perturbation`이 계측 비용이다(8B, 2 GPU, 4k에서 +8.5%). `acceptance_identical`이 True여야 계측이 실행을 바꾸지 않은 것이다.

#### 3.4.3 OOM 기록

어느 패스든 OOM이 나면 `median.allocator_at_oom`에 장치별로 다음 값을 남긴다.

- `allocated_bytes`, `reserved_bytes`
- `inactive_split_bytes`(단편화)
- `num_alloc_retries`
- `device_free_bytes`, `device_total_bytes`

`reserved − allocated`가 크고 `inactive_split_bytes`가 크면 단편화 때문이다. `device_free`가 작고 allocated가 거의 전부라면 용량 한계다. 어느 장치에서 났는지는 에러 문자열(`GPU 1 has a total capacity...`)과 이 스냅샷으로 판단한다.

### 3.5 acceptance와 파생 지표

| 필드 | 방식 |
|---|---|
| `acceptance_rate` | 수락된 draft 토큰 ÷ 제안된 draft 토큰 |
| `mean_committed_per_step` | 스텝당 확정 토큰(수락 + 보정/보너스 1) |
| `num_verify_steps`, `committed_per_step` | 스텝 수와 스텝별 확정 수 |
| `speedup.<p>.tpot_speedup` | AR TPOT ÷ DFlash TPOT(분모 모두 n−1) |
| `speedup.<p>.verify_over_ar_step` | `target_verify_mean_s` ÷ AR TPOT. **acceptance와 무관한** 스텝당 비용 비 |
| `speedup.<p>.peak_memory_sum_device_maxima_bytes_overhead` | DFlash perf − AR, 둘 다 장치별 최댓값 합. drafter 비중 질문의 분자 |
| `speedup.<p>.ar_drafter_resident` | False여야 한다 |

acceptance는 장치 배치와 무관해야 하지만 GDN 커널 백엔드에는 영향을 받는다. 그래서 `gdn_backend`가 다른 레코드끼리는 합치지 않는다.

### 3.6 trace 패스: 장치 busy/idle과 verify 분해

실행 방식:

- perf 패스 뒤에 1회 실행한다. perf 패스를 돌리지 않았다면 warmup 1회를 먼저 한다.
- **프로파일 창.** profiler는 step phase(DFlash는 `decode: target verify`, AR은 `decode: ar step`)가 처음 열릴 때 시작하고, `--trace-steps`(기본 32)번째 스텝이 끝나면 멈춘다. 그래서 prefill과 drafter의 첫 호출(프롬프트 길이 context 투영)은 트레이스에 들어가지 않는다. 멈추기 전에 모든 장치를 synchronize해서, 먼 shard에서 실행된 마지막 스텝의 커널까지 창에 들어가게 한다.
  - 전체 실행을 프로파일하지 않는 이유: 35B-A3B는 HF의 참조 MoE 구현이 expert마다 커널을 launch한다. S=4k AR 한 번의 전체 트레이스가 gzip으로 1 GB였고, 파싱에 호스트 메모리를 100 GB 넘게 썼다.
  - 창은 decode의 **앞쪽** 스텝들이다. 스텝당 비용은 context가 길어지면서 조금씩 변하므로, 창의 값은 decode 전체의 평균이 아니다.
- `torch.profiler`(CPU와 CUDA activity, CUPTI)로 실행한다. 메모리 추적과 층 훅은 끈다.
- 두 종류의 호스트 range를 건다.
  - **phase range:** `phase/<label>`. `dflash_generate`와 `ar.generate`가 phase 경계마다 호출한다.
  - **module range:** `mod/target/L<i>/<kind>`. 각 층의 mixer(`full_attention`/`gdn`), FFN(`ffn_dense`/`ffn_moe+shared`), MoE 하위 모듈(`moe_router`, `moe_experts`, `moe_shared_expert`), `lm_head`에 forward 훅으로 건다.
- 원본 트레이스는 `traces/.../*.json.gz`로 저장한다. Perfetto나 chrome://tracing에서 열 수 있다.

분석(`trace.analyse`):

1. **귀속.** 각 GPU 이벤트(kernel, memcpy, memset)를 CUPTI correlation id로 그 이벤트를 launch한 호스트 호출(`cudaLaunchKernel`/`cuLaunchKernel`)에 연결한다. launch 시각을 감싸는 가장 안쪽의 phase range와 module range에 귀속시킨다. 어느 장치에서 실행되었든 **launch한 코드 위치**로 귀속되므로, `cuda:1`에 있는 층의 커널은 그 층의 module에 들어간다.
2. **창.** decode 창은 프로파일한 구간에서 decode phase가 launch한 첫 GPU 이벤트의 시작부터 마지막 이벤트의 끝까지다. `analysis.profiled_steps`와 `window_s_per_step`이 함께 기록된다.
3. **장치 판정.** kernel은 `args.device`, memcpy와 memset은 `args.inDevice`(복사를 실행한 엔진의 장치)로 장치를 정한다. P2P 복사(`Memcpy PtoP`)는 보통 **송신 측 장치**에 잡힌다.

장치별 출력 `analysis.devices["cuda:i"]`(CSV `trace_device.csv`):

| 필드 | 계산 | 해석 |
|---|---|---|
| `busy_s`, `busy_fraction` | 창 안의 kernel, memcpy, memset 구간의 **합집합** ÷ 창 | 이 장치에 무엇이든 실행 중이던 시간의 비율. SM 점유율이나 대역폭 사용률이 **아니다** |
| `kernel_busy_s` | 커널만의 합집합 | |
| `memcpy_s`, `p2p_memcpy_s`, `p2p_memcpy_bytes`, `memcpy_count` | 이 장치 엔진이 실행한 복사 | 장치 경계 전송 비용(B5) |
| `idle_s` | 창 − busy | |
| `idle_shard_wait_s` | idle 중 **다른 장치가 busy였던** 시간 | layer 샤딩의 구조적 대기, 그리고 drafter(`cuda:0`)가 도는 동안 `cuda:1`의 대기 |
| `idle_all_devices_s` | **어느 장치도** busy가 아니었던 시간(모든 장치에 같은 값) | 호스트 launch 공백, `.item()` 동기화, Python 오버헤드 |

phase별 출력 `analysis.phases[<phase>]`(CSV `trace_phase.csv`, `trace_kernel.csv`):

| 필드 | 계산 |
|---|---|
| `occurrences` | phase 진입 횟수(verify 수, AR 스텝 수) |
| `device_time_s`, `device_time_per_occurrence_s` | 귀속된 GPU 이벤트 시간의 **합**(장치 합산) |
| `device_s` | 장치별 합 |
| `module_kind_s` | 모듈 종류별 합. `outside_target_modules`에는 drafter 커널, 장치 간 복사, argmax 등 target 모듈 밖에서 launch된 것이 들어간다 |
| `layer_s` | 층별 합 |
| `kernel_category_s`, `top_kernels` | 커널 이름을 정규식으로 분류(`attention_flash`, `attention_mem_efficient`, `gemm`, `gdn_or_conv`, `elementwise` 등). 분류는 이름에 기반한 휴리스틱이고, 1차 분해는 module 기준이다 |

`attention_calls`(CSV `attention_calls.csv`): full-attention 모듈 호출마다 (phase, query 토큰 수)별로 첫 호출의 `attention_mask` shape과 dtype, `_attn_implementation`, 호출 수를 기록한다. SDPA가 어떤 커널을 고를지를 결정하는 입력이다.

해석상 주의할 점:

- **device_time은 합이고 busy는 합집합이다.** 한 장치 안에서는 같은 스트림의 커널이 직렬로 실행되므로 둘이 거의 같다. 그러나 장치를 합산한 `device_time_s`는 장치들이 병렬로 일했다면 벽시계보다 클 수 있다. B=1 샤딩에서는 거의 직렬이다.
- **profiler가 호스트 시간을 부풀린다.** `trace_perturbation.{dflash,ar}.ratio` = (창 길이 ÷ 프로파일한 스텝 수) ÷ (perf의 스텝당 시간)이다. 8B 4k에서 DFlash 1.42, AR 1.76이었다. 이만큼 스텝이 느려지며, 늘어난 시간은 주로 `idle_all_devices_s`로 간다. 커널 실행 시간 자체는 부풀지 않는다. 그래서 커널과 모듈의 [dev] 시간은 믿을 수 있고, idle 비율은 상한으로 읽는다.
- **장치 간 시각 정렬:** CUPTI가 각 장치의 타임스탬프를 공통 시간축으로 변환한다. 장치 간 us 단위 오차가 있을 수 있으며, 이 오차는 검증하지 않았다. shard wait와 all-idle 판정은 이 정렬에 의존한다.
- **하드웨어 카운터(DRAM 바이트, SM throughput)는 없다.** 설치된 ncu가 이 torch 빌드에서 동작하지 않는다.

### 3.7 moe 패스: 라우팅

- AR 1회(native)와 DFlash 1회를 실행하며, MoE 층마다 router(`mlp.gate`)에 forward 훅을 건다.
- 훅은 top-k expert id를 `bincount`하고 CPU로 복사한다. 이 복사는 **층마다 그 층이 있는 장치를 동기화**하므로 시간 값은 기록하지 않는다(`time_parts=False`).
- 시간이 필요하면 trace 패스의 `moe_router`, `moe_experts`, `moe_shared_expert` 모듈 시간을 쓴다.

| 필드(`moe_routing.csv`, 층 × forward당 1행) | 의미 |
|---|---|
| `path`, `phase`, `step` | ar/dflash, phase 이름, AR 스텝 또는 verify 번호 |
| `layer`, `device` | 층 번호와 그 층의 장치(taxonomy 기준) |
| `num_tokens`, `top_k`, `num_experts` | 이 forward에서 라우팅된 토큰 수(AR 1, verify는 블록 폭) |
| `unique_experts_hit`, `expert_hit_fraction` | 실제로 로드해야 했던 expert 수와 비율 |
| `tokens_per_expert_max/mean`, `load_imbalance` | expert 루프의 꼬리를 결정하는 불균형 |

JSON에는 decode 구간의 토큰별 top-k id(`top_k_expert_ids`)도 남는다. prefill은 토큰이 256개를 넘으므로 bincount 요약만 남는다.

### 3.8 verify-width (`verify_width_<N>gpu_*.json`)

- **입력:**
  - fixed-task 프롬프트 1개.
  - 검증 블록: prefill의 argmax 토큰 뒤에 target의 greedy 연속 토큰을 이어 붙인 것.
  - 캐시 정책별(native, recording)로 같은 블록을 쓴다.
- **측정:** 같은 캐시 상태로 복원한 뒤 q ∈ {1,4,8,16} 폭의 target forward를 `cuda:0` synchronize 사이에서 잰다 [wall]. 반복의 중앙값을 쓴다.
- **멀티 GPU:** 시간에는 장치 두 개와 경계 복사가 모두 포함된다. 복원 비용(`mean_restore_s`)은 따로 보고한다.
- native 정책의 q=1 행이 AR 한 스텝에 해당한다. recording 정책의 q=1은 다른 conv 커널 경로를 탄다.

### 3.9 lossless (audit/exact 패스, 선택 사항)

`MEASUREMENT_AUDIT.md` D절과 같다. 각 패스는 1회만 실행하며 시간 값은 쓰지 않는다. 출력 토큰 비교(`stock_vs_ar` 등)와 GDN 상태 오차가 결과다.

---

## 4. 비교 규칙

| 비교 | 가능 여부 | 이유 |
|---|---|---|
| DFlash TPOT 대 AR TPOT(같은 레코드, 같은 조건) | 가능 | 같은 배치, 같은 분모, 둘 다 계측 없음 |
| `*_sum_device_maxima_bytes` DFlash 대 AR | 가능(차이만) | 같은 정의. 절대값은 동시 피크보다 과대 |
| `peak_memory_simultaneous_bytes` 대 AR 값 | **불가** | AR에는 구간 분해 값이 없다 |
| `..._at_aggregate_peak_bytes`[i] 대 다른 값의 [i] | **불가** | 한 순간의 분해일 뿐 장치 i의 피크가 아니다 |
| perf 대 memory 패스의 시간 | 계측 비용을 읽을 때만 | memory 패스에는 층 훅과 allocator 읽기가 있다 |
| trace의 idle 비율 대 perf 시간 | 상한으로만 | profiler가 호스트 시간을 부풀린다 |
| trace의 모듈/커널 [dev] 시간끼리(AR 대 verify) | 가능 | 커널 시간은 profiler 영향을 거의 받지 않는다 |
| N GPU 레코드 대 다른 N | 절대 시간과 메모리는 **불가** | 경계 수, 층 분할, 장치당 가중치가 다르다. 비율(speedup)은 참고로만 |
| v1 레코드 대 v2 | 메모리, TPOT **불가** | v1은 AR에 drafter가 상주했고, TPOT 분모와 perf 계측도 다르다 |
| `gdn_backend`가 다른 레코드 | **불가** | acceptance 자체가 달라진다 |
| 다른 S 사이의 acceptance(fixed-task) | 조건부 | 같은 원본 문서지만 컨텍스트 양이 다르다. K개 프롬프트의 분산과 함께 본다 |

---

## 5. CSV 요약

| 파일 | 행 단위 | 출처 패스 | 핵심 열 |
|---|---|---|---|
| `speedup.csv` | 조건 | perf, memory, AR | TPOT, acceptance, speedup, `verify_over_ar_*_step`, 메모리(같은 정의끼리), AR `drafter_resident` |
| `timing.csv` | 조건 | perf | 구간별 total과 mean, 각 항목의 decode 대비 비율, unattributed |
| `device.csv` | 조건 × 장치 | memory, perf, AR | 세 가지 정의의 장치별 메모리를 분리된 열로 |
| `phase.csv` | 조건 × phase | memory | phase별 구간 피크와 그 시점의 구성 요소 |
| `component.csv` | 조건 × 시점(prefill 끝, 첫 draft, steady) | memory | target KV, GDN 상태, draft KV, 비율 |
| `trace_device.csv` | 조건 × 경로 × 장치 | trace | busy, shard wait, all-idle, P2P |
| `trace_phase.csv` | 조건 × 경로 × phase × 분해 항목 | trace | 모듈 종류별, 커널 분류별 [dev] 시간과 발생당 시간 |
| `trace_kernel.csv` | 조건 × 경로 × phase × 순위 | trace | 상위 커널 이름과 시간 |
| `attention_calls.csv` | 조건 × 경로 × phase × q | trace | 마스크 shape, attn 구현 |
| `moe_routing.csv` | 조건 × 경로 × forward × 층 | moe | hit expert 수, 불균형 |
| `lossless.csv` | 조건 × 참조 | audit, exact | 상태 오차, KL, argmax 뒤집힘 |
| `verify_width.csv` | S × 정책 × q | verify-width 레코드를 export한 경우 | q별 비용, q=1 대비 비율 |

CSV는 JSON에서 파생된 표이며, 측정을 다시 계산하지 않는다. 값이 없으면 `NA`로 남긴다. 언제든 `python -m dflash.arch export <record.json>`으로 다시 만들 수 있다.

---

## 6. 첫 v2 기록: Qwen3.5-35B-A3B, 2 GPU, sequence sweep

- 기록: `record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_2gpu_20260929-192934.json`
- CSV: `csv/sweep_causal_conv1d+fla_2gpu_20260929-192934/`
- 조건: 2wikimqa, source_index 66(모든 S에서 같은 문서, 8k 이상은 합성 확장), block 16, 256 토큰(shape-controlled), 반복 3, 패스 perf, memory, moe, trace(32 스텝)
- 배치: 20/20층, 가중치 35.55 / 34.66 GB, drafter 0.77 GB(`cuda:0`)
- 모든 시간 측정 행에서 `profiler_used_before=False`이고, AR 행은 `drafter_resident=False`다.

### 6.1 지연 시간과 acceptance

| S | AR native | AR recording | DFlash perf | acceptance | 확정 토큰/스텝 | speedup (native 대비) | verify 1회 / AR 1스텝 |
|---|---|---|---|---|---|---|---|
| 4k | 129.8 ms | 147.9 ms (±12.1) | 28.9 ms | 0.402 | 6.71 | 4.49× | 183 / 130 = 1.41 |
| 8k | 131.2 | 154.0 (±13.8) | 39.5 | 0.270 | 5.00 | 3.32× | 186 / 131 = 1.42 |
| 16k | 130.3 | 149.3 (±6.2) | 34.5 | 0.349 | 6.07 | 3.78× | 197 / 130 = 1.51 |
| 32k | 130.6 | OOM | OOM | — | — | — | — |
| 64k | 131.0 | OOM | OOM | — | — | — | — |

- AR native는 S와 무관하게 약 130 ms다. S축에서 speedup이 바뀌는 것은 acceptance(0.40 → 0.27 → 0.35) 때문이고, verify 비용은 1.41 → 1.51로 조금만 늘었다. 프롬프트가 S당 1개(K=1)라서 acceptance의 프롬프트 간 분산은 아직 측정하지 않았다.
- v1(3-GPU) 기록의 16k acceptance 0.091은 trec 과제에서 나온 값이었다. 과제를 고정하자 0.349가 나왔으므로, 그 값은 과제 교란 때문이었음이 확인된다. 단, 배치가 달라 두 기록을 수치로 직접 비교하지는 않는다.
- AR recording은 native보다 14–18% 느리고 편차가 크다(±6–14 ms). 9B에서 측정한 약 2%보다 훨씬 크며, 원인은 이번 기록으로 분해되지 않았다.
- memory 패스의 계측 비용은 +1.9–2.7%이고, 모든 조건에서 `acceptance_identical=True`였다.

### 6.2 decode 한 스텝의 구성 (trace, 창 32 스텝의 발생당 값)

| | AR 1스텝 (4k) | verify 1회 (4k) | verify 1회 (8k) |
|---|---|---|---|
| 벽시계(perf) | 130 ms | 183 ms | 186 ms |
| **GPU 장치 시간(두 장치 합)** | **18.9 ms** | **57.2 ms** | 63.4 ms |
| kernel launch 수 | 4,464 | 9,239 | 9,347 |
| moe_experts | 6.3 ms | 33.1 ms | 33.9 ms |
| gdn | 5.1 | 8.1 | 8.2 |
| full_attention | 2.2 (flash) | 6.6 (mem-efficient + 마스크) | 12.4 |
| lm_head | 1.4 | 1.4 | 1.4 |
| 층당 hit expert 수(256개 중) | 8.0 | 평균 48.5, 최대 100 | 49.0 |

- **35B의 decode는 호스트 launch가 병목이다.** AR 한 스텝 130 ms 중 GPU가 일한 시간은 두 장치를 합쳐 19 ms(약 15%)뿐이다. 나머지는 어느 GPU도 일하지 않는 시간이다. 스텝마다 4,464개의 커널이 launch되며, 이는 HF 참조 MoE 구현이 hit expert마다 Python 루프를 돌기 때문이다. 커널당 약 29 µs의 호스트 시간이 벽시계를 결정한다. 감사 보고서 B3에서 "35B AR 135 ms/token의 원인"으로 남겨 두었던 질문의 답이다.
  - trace의 busy 비율(장치당 약 3%)은 profiler가 스텝을 2.1–2.6배 늘린 상태의 값이다. 그래서 위 판단은 profiler에 영향받지 않는 커널 시간과 perf 벽시계를 조합해서 내렸다.
- **verify가 AR 스텝보다 1.4배 비싼 주원인은 MoE다.** 16토큰이 층마다 평균 48.5개의 expert를 건드린다(AR은 8개). 그래서 expert GEMM 시간이 5.2배, launch 수가 2.1배가 된다. 벽시계 비율(1.41)은 GPU 시간 비율(3.0)보다 launch 수 비율에 가깝다. verify도 호스트가 병목이기 때문이다.
- **어텐션:** 8B와 같은 현상이 보인다. verify는 bool 마스크 때문에 mem-efficient 커널을 쓰고, AR은 flash를 쓴다. 어텐션 층이 10개뿐이어서 4k에서는 비중이 작지만, 4k → 8k에서 6.6 → 12.4 ms로 S에 비례해 늘어난다.
- **drafter** 스텝당 비용은 draft forward 4.6 ms(`cuda:0`, 약 500 커널)와 draft logits 4.4 ms(`cuda:1`의 lm_head)다. `cuda:1`에서 출발하는 P2P 복사가 스텝당 약 6 ms로, AR(0.1 ms)보다 훨씬 크다. draft logits와 verify logits를 되돌려 보내는 복사다.

### 6.3 메모리와 2-GPU 용량 한계

| S | AR native 장치별 최댓값 | DFlash perf 장치별 최댓값 | DFlash 동시 피크(memory 패스) | 오버헤드: native 대비 / recording 대비 |
|---|---|---|---|---|
| 4k | 36.3 / 35.4 GB | 38.1 / 36.4 | 73.9 (피크 위치: `target layer 39`) | 2.85 / 0.84 GB |
| 8k | 36.9 / 36.0 | 39.9 / 38.0 | 76.8 | 4.93 / 0.91 |
| 16k | 38.3 / 37.3 | 43.3 / 41.3 | 82.5 | 9.09 / 1.04 |

- DFlash의 메모리 오버헤드 대부분은 drafter가 아니라 **recording 정책(GDN conv recording buffer, O(S))** 에서 온다. recording AR 대비로 보면 drafter 자체의 몫(가중치 0.77 GB, draft KV, context feature)은 1 GB 안팎이다.
- **32k와 64k OOM**
  - DFlash: `cuda:0`에서 48.6–49.6 GB가 할당된 상태에서 실패했다. reserved인데 쓰이지 않은 메모리는 0.5–0.8 GB뿐이므로 용량 한계다. `cuda:1`은 가중치 34.67 GB만 가진 상태였으므로 prefill의 첫 shard에서 실패한 것이다.
  - AR recording도 32k와 64k에서 OOM이 났고, 32k에서는 `inactive_split` 6.15 GB가 기록되었다. 단편화가 실제로 기여했다는 뜻이다.
  - AR native는 64k까지 성공했다(`cuda:0` 최댓값 46.3 GB).
  - 따라서 2-GPU 35B에서 32k 이상을 측정하려면 recording buffer를 줄이는 수정이 필요하다. 예를 들어 prefill이 끝난 직후 crop해서 버퍼를 해제하는 방법이 있다. 이는 측정이 아니라 DFlash 구현을 바꾸는 일이다.
- 진단 패스(moe, trace)는 drafter가 상주한 상태에서 돌기 때문에 16k부터 OOM이 났다(DFlash moe와 trace는 16k, AR moe와 trace는 32k부터). 16k 이상의 MoE 라우팅과 trace 분해는 이 기록에 없다.

---

## 7. Qwen3.5-35B-A3B, 3 GPU, sequence sweep

- 기록: `record_arch_main/qwen3.5-35b-a3b/sweep_causal_conv1d+fla_3gpu_20260930-023904.json`
- 조건: 6절(2 GPU)과 같다. 같은 source 문서, block 16, 256 토큰, 반복 3, 패스 perf/memory/moe/trace.
- 배치: 13/13/14층, 가중치 23.78/21.87/24.56 GB.
- 모든 시간 측정 행에서 `profiler_used_before=False`, AR 행에서 `drafter_resident=False`다.

### 7.1 지연 시간

| S | AR native | AR recording | DFlash | acceptance | speedup(native) | verify 1회 | verify ÷ AR 스텝 | 2 GPU DFlash |
|---|---|---|---|---|---|---|---|---|
| 4k | 131.7 ms | 144.0 | 28.7 ms | 0.402 | 4.59× | 183 ms | 1.39 | 28.9 |
| 8k | 131.1 | 152.8 | 38.9 | 0.270 | 3.37× | 185 | 1.41 | 39.5 |
| 16k | 131.4 | 145.8 | 33.8 | 0.349 | 3.89× | 195 | 1.49 | 34.5 |
| 32k | 132.1 | 151.3 | 35.9 | 0.368 | 3.67× | 218 | 1.65 | OOM |
| 64k | 132.2 | **OOM** | 38.4 | 0.417 | 3.45× | 257 | 1.95 | OOM |

- 4k–16k에서 2 GPU와 3 GPU의 값은 1–2% 안에서 같다. decode는 호스트 launch가 병목이라(6.2절), GPU를 한 장 늘려도 층 경계 하나가 더 생길 뿐 시간은 거의 변하지 않는다.
- 같은 문서, 같은 acceptance(4k–16k 동일)이므로 2 GPU와 3 GPU의 차이는 배치의 효과뿐이다. 이 비교는 4절의 "다른 N 사이 절대값 비교 불가" 규칙에 대한 예외적 확인이다.

### 7.2 verify 비용이 S에 따라 커지는 이유: full attention

verify 1회의 장치 시간(trace, 32 스텝 평균):

| S | verify 장치 시간 | full_attention | moe_experts | gdn | AR 스텝의 full_attention(flash) |
|---|---|---|---|---|---|
| 4k | 54.4 ms | 6.5 | 33.0 | 8.1 | 2.2 |
| 8k | 61.2 | 12.4 | 33.7 | 8.1 | 2.5 |
| 16k | 69.8 | 22.8 | 31.7 | 8.1 | 3.4 |
| 32k | 91.3 | 42.6 | 33.3 | 8.1 | 4.8 |
| 64k | 132.7 | **83.2** | 34.2 | 8.1 | 7.5 |

- MoE expert 비용은 S와 무관하게 약 33 ms다. 층당 hit expert는 47.5–51.5개(AR은 8개)다.
- verify의 full attention은 S에 비례해 늘어, 64k에서 verify 장치 시간의 63%를 차지한다. 같은 S에서 AR의 flash 어텐션보다 11배 비싸다. 원인은 8B에서 확인한 것과 같다. verify는 bool 마스크 `[1,1,16,S]`를 materialize해서 mem-efficient 커널과 elementwise 연산을 쓰고, AR은 마스크 없이 flash를 쓴다. 어텐션 층이 10개뿐인 하이브리드 모델에서도 64k에서는 이것이 verify의 최대 비용이다.
- verify ÷ AR 스텝 비가 1.39에서 1.95로 오르는 것도 이 때문이다.

### 7.3 메모리

| S | AR native 장치별 최댓값 | DFlash 장치별 최댓값 | DFlash 동시 피크 | 오버헤드(native 대비 / recording 대비) |
|---|---|---|---|---|
| 4k | 24.5 / 22.5 / 25.2 GB | 25.9 / 23.2 / 25.9 | 73.9 | 2.82 / 0.81 |
| 16k | 26.4 / 24.4 / 27.2 | 30.0 / 27.1 / 29.8 | 82.5 | 8.96 / 0.91 |
| 32k | 29.0 / 27.0 / 29.7 | 35.4 / 32.3 / 35.1 | 93.9 | 17.14 / 1.04 |
| 64k | 34.2 / 32.0 / 34.9 | 46.3 / 42.8 / 45.6 | 116.8 | 33.52 / — |

- drafter 자체의 몫(recording 대비 오버헤드)은 S와 관계없이 약 1 GB다. native 대비 오버헤드의 나머지는 recording 정책(GDN conv recording buffer)이며 S에 비례한다.
- **64k AR recording의 OOM은 용량이 아니라 단편화 때문이다.** 세 장치 모두에서 reserved가 약 50 GB인데 allocated는 33–36 GB이고, `inactive_split_bytes`가 13.1–14.0 GB다. 같은 S에서 DFlash는 장치당 최대 46 GB로 성공했다. AR recording 루프가 매 스텝 conv 버퍼를 crop(슬라이스)하면서 할당자를 조각낸 것으로 보인다. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 재측정하면 확인할 수 있다(검증하지 않음).
- 2 GPU의 32k/64k OOM은 이와 달리 용량 한계였다(6.3절, reserved-unallocated 1 GB 미만).

### 7.4 기타 관찰

- draft logits는 스텝당 1.9 ms로, 2 GPU(4.1 ms)보다 싸다. 원인은 분해하지 않았다.
- memory 패스의 계측 비용은 +1.7–3.0%였고 acceptance는 전 조건에서 동일했다. profiler 섭동은 DFlash 1.85–2.22배, AR 2.61배였다.
