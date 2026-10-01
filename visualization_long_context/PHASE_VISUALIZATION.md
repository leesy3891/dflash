# 전체 context의 phase 자원 및 latency 분석

세 모델 각각의 4K·8K·16K·32K·64K를 모두 표시한다. 기본 block은 16, 출력은 256, B=1이며 같은 3-GPU protocol v2 기록을 사용한다.

## 모델별 그림

| 모델 | Phase 메모리 | DFlash / native AR latency | Verify / AR 내부 모듈 비용 |
| --- | --- | --- | --- |
| qwen3-8b | [PNG](phase_memory_qwen3-8b.png) / [PDF](phase_memory_qwen3-8b.pdf) | [PNG](phase_latency_qwen3-8b.png) / [PDF](phase_latency_qwen3-8b.pdf) | [PNG](target_execution_qwen3-8b.png) / [PDF](target_execution_qwen3-8b.pdf) |
| qwen3.5-9b | [PNG](phase_memory_qwen3.5-9b.png) / [PDF](phase_memory_qwen3.5-9b.pdf) | [PNG](phase_latency_qwen3.5-9b.png) / [PDF](phase_latency_qwen3.5-9b.pdf) | [PNG](target_execution_qwen3.5-9b.png) / [PDF](target_execution_qwen3.5-9b.pdf) |
| qwen3.5-35b-a3b | [PNG](phase_memory_qwen3.5-35b-a3b.png) / [PDF](phase_memory_qwen3.5-35b-a3b.pdf) | [PNG](phase_latency_qwen3.5-35b-a3b.png) / [PDF](phase_latency_qwen3.5-35b-a3b.pdf) | [PNG](target_execution_qwen3.5-35b-a3b.png) / [PDF](target_execution_qwen3.5-35b-a3b.pdf) |

## D4 verify와 steady draft의 상세 그림

- [D4 verify 메모리 split: 3모델 × 5context](verify_memory_split_all_contexts.png) / [PDF](verify_memory_split_all_contexts.pdf)
- [D4 verify overhead: AR 대비 호출 비용, 출력 토큰당 비용, latency 기여율](verify_overhead_all_contexts.png) / [PDF](verify_overhead_all_contexts.pdf)
- [Steady draft의 커널별 시간 및 비율](steady_draft_all_contexts.png) / [PDF](steady_draft_all_contexts.pdf)
- [First / steady draft, draft logits, verify, AR의 호출당 시간](phase_calls_all_contexts.png) / [PDF](phase_calls_all_contexts.pdf)

## 어떻게 읽는가

Phase 메모리 그림은 각 context마다 전체 메모리, target/draft 가중치를 제외한 구성, native AR의 전체 실행 peak를 나란히 표시한다. 같은 모델에서는 context별 y축 범위가 같아서 자원 증가량을 비교할 수 있다. 각 phase의 반복 중 가장 큰 occurrence를 사용하므로 연결된 점은 단일 iteration의 시간 순서가 아니다. First draft 옆의 steady draft도 비교를 위한 배치다.

D4 split은 같은 메모리 측정에서 verify phase만 추출한다. 보라색 사선 영역은 interval peak에서 phase 종료시 probe된 구성 요소를 뺀 나머지다. Verify 도중 해제된 prompt conv recording도 여기에 들어갈 수 있으므로 전부 attention activation이라고 해석하지 않는다. 가중치 제외 그림은 weights를 제거한 시각적 분해이며, AR 대비 메모리 overhead를 뜻하지 않는다.

Latency 그림은 DFlash와 해당 모델의 native AR을 context별로 한 쌍씩 표시한다. 상단은 TTFT+decode 전체 request, 하단은 decode만의 절대 시간과 비율이다. First draft 1회와 steady draft 전체 호출을 구분한다. Prefill의 target forward/첫 토큰/기타는 개별 wall timer가 없어서 하나의 항으로 남긴다. Crop/동기화/기타 bookkeeping도 decode remainder에 남긴다.

Target 모듈 그래프는 trace의 `module_kind_s`를 occurrence 수로 나눈 장치 activity다. **커널과 memcpy를 모두 포함**하며, attention mixer/FFN/GDN/LM head 등의 시간과 비율을 표시한다. 이 비율의 분모는 장치 activity이므로 request wall latency 기여율과 다르다. Black tick은 별도 perf 패스의 verify 또는 AR interval이다. 차이를 CPU 시간이나 GPU utilization으로 해석하지 않는다.

Steady draft 내부 그래프는 `kernel_category_s`를 이용한 **커널만의 시간**이다. GEMM에는 여러 projection과 MLP가 포함되며 특정 projection 하나로 귀속하지 않는다. Profiler window는 first draft를 포함하지 않는다.

Verify overhead 그림의 왼쪽은 block verify 1회와 AR 1토큰 step을 비교한다. `+ms`는 두 호출의 차이이고, 같은 토큰 수의 작업을 뺀 순수 overhead 측정이 아니다. 가운데는 verify 총 시간 / 실제 decode 출력 토큰 수(255)이므로 acceptance와 반복 횟수가 반영된다. 오른쪽은 verify의 decode 기여율과 E2E 기여율을 구분한다.

## 미측정 항목

이번 arch 기록에는 **native AR phase별 메모리 probe와 draft의 의미적 5-stage timer 값이 저장되어 있지 않다**. 따라서 AR phase별 KV/activation split, first/steady draft의 fc+norm/context-KV/cache append 단계별 wall time을 정확하게 재구성할 수 없다. 그림에는 AR의 실제 전체 실행 allocated/reserved peak와 가중치만 표시했고, draft 내부는 저장된 trace 커널 구성으로 표시했다. 이전 selective sweep은 모델·GPU 배치·측정 조건이 달라 이번 결과와 혼합하지 않았다.

Hybrid 측정은 stock rollback 경로다. 정확한 state restoration 비용을 포함한 lossless speedup은 미확정이며, 64K의 8B RoPE 범위 및 35B draft 학습 길이 제한도 유지된다.

## 작은 phase까지 확인하는 정량 표

아래는 막대 안에 글자가 들어가지 않는 first/steady draft도 정확히 읽을 수 있도록 제공하는 표다. `steady 전체`는 한 호출의 시간이 아니라 해당 request에서 반복된 모든 steady forward의 합이다. 모든 phase의 E2E/decode 비율은 CSV에 별도로 저장한다.

### qwen3-8b

| Context | DF TTFT / AR TTFT (s) | First draft (ms) | Steady 전체 (s) / 호출 평균 (ms) | Draft logits 전체 (s) | Verify 전체 (s) / 호출 평균 (ms) | Verify % decode / E2E | DF decode / AR decode (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 0.704 / 0.692 | 15.714 | 0.465 / 5.334 | 0.176 | 4.206 / 47.797 | 85.67% / 74.92% | 4.910 / 9.613 |
| 8K | 1.451 / 1.452 | 26.697 | 0.585 / 5.892 | 0.200 | 6.306 / 63.055 | 87.95% / 73.14% | 7.170 / 9.729 |
| 16K | 3.390 / 3.304 | 49.217 | 0.778 / 6.867 | 0.228 | 11.795 / 103.464 | 91.36% / 72.36% | 12.910 / 10.167 |
| 32K | 8.274 / 8.200 | 96.343 | 1.527 / 8.964 | 0.342 | 29.306 / 171.378 | 93.45% / 73.94% | 31.360 / 12.665 |
| 64K | 23.589 / 23.362 | 187.657 | 2.643 / 13.270 | 0.399 | 61.294 / 306.469 | 94.83% / 69.48% | 64.633 / 17.457 |

### qwen3.5-9b

| Context | DF TTFT / AR TTFT (s) | First draft (ms) | Steady 전체 (s) / 호출 평균 (ms) | Draft logits 전체 (s) | Verify 전체 (s) / 호출 평균 (ms) | Verify % decode / E2E | DF decode / AR decode (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 0.688 / 0.679 | 23.708 | 0.221 / 7.872 | 0.094 | 1.724 / 59.458 | 83.09% / 62.41% | 2.075 / 10.717 |
| 8K | 1.359 / 1.353 | 39.146 | 0.444 / 8.059 | 0.181 | 3.359 / 59.984 | 82.95% / 62.10% | 4.050 / 10.706 |
| 16K | 2.820 / 2.782 | 76.347 | 0.203 / 8.124 | 0.084 | 1.562 / 60.092 | 80.65% / 32.84% | 1.937 / 10.696 |
| 32K | 6.002 / 5.839 | 145.688 | 0.624 / 8.652 | 0.236 | 5.038 / 69.020 | 82.91% / 41.71% | 6.077 / 10.592 |
| 64K | 13.521 / 13.207 | 286.525 | 0.238 / 9.494 | 0.084 | 2.623 / 100.882 | 80.88% / 15.65% | 3.243 / 10.680 |

### qwen3.5-35b-a3b

| Context | DF TTFT / AR TTFT (s) | First draft (ms) | Steady 전체 (s) / 호출 평균 (ms) | Draft logits 전체 (s) | Verify 전체 (s) / 호출 평균 (ms) | Verify % decode / E2E | DF decode / AR decode (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 0.708 / 0.698 | 11.581 | 0.243 / 6.533 | 0.073 | 6.971 / 183.435 | 95.23% / 86.83% | 7.320 / 33.595 |
| 8K | 1.201 / 1.177 | 20.166 | 0.322 / 6.419 | 0.099 | 9.442 / 185.135 | 95.27% / 84.97% | 9.911 / 33.425 |
| 16K | 2.282 / 2.246 | 38.335 | 0.275 / 6.691 | 0.078 | 8.197 / 195.169 | 95.22% / 75.27% | 8.608 / 33.503 |
| 32K | 4.801 / 4.734 | 74.807 | 0.280 / 7.154 | 0.075 | 8.716 / 217.901 | 95.11% / 62.41% | 9.164 / 33.677 |
| 64K | 11.107 / 10.957 | 148.579 | 0.282 / 8.039 | 0.067 | 9.263 / 257.319 | 94.74% / 44.35% | 9.778 / 33.700 |

## 데이터와 재현

| CSV | 범위 |
| --- | --- |
| [phase_memory_all_contexts.csv](phase_memory_all_contexts.csv) | 180행: 3모델 × 5context × 12phase. GB 구성, before/after/peak, occurrence, borrowed |
| [ar_memory_all_contexts.csv](ar_memory_all_contexts.csv) | 15행: native AR weights 및 전체 실행 allocated/reserved sums of device maxima |
| [phase_latency_all_contexts.csv](phase_latency_all_contexts.csv) | 30행: 3모델 × 5context × DFlash/AR. 각 phase seconds, E2E 및 decode 비율 |
| [phase_call_latency_all_contexts.csv](phase_call_latency_all_contexts.csv) | 15행: 호출당 ms 및 호출 수 |
| [target_module_latency_all_contexts.csv](target_module_latency_all_contexts.csv) | Verify/AR 모듈별 device activity ms 및 비율; kernels + memcpy |
| [steady_draft_kernel_latency_all_contexts.csv](steady_draft_kernel_latency_all_contexts.csv) | Steady draft kernel별 ms 및 비율; kernels only |
| [verify_memory_all_contexts.csv](verify_memory_all_contexts.csv) | 15행: D4 memory split |
| [verify_overhead_all_contexts.csv](verify_overhead_all_contexts.csv) | 15행: verify/AR 비용 차, 비율, 확정 출력당 비용, decode/E2E 기여율 |

Trace CSV의 `part_ms_per_occurrence`, `part_share_pct`, `total_profiled_ms_per_occurrence`는 `activity_kind`에 따라 kernels-only 또는 kernels+memcpy를 뜻한다.

```bash
/home/seoyounglee/venvs/dflash-fla/bin/python visualization_long_context/plot_phases.py
```

[phase_sources.json](phase_sources.json)에 입력 기록을 고정했다. 새 GPU 실험 없이 기존 데이터로 재현한다.
