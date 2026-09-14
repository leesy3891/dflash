# visualization_batch

`record_batch/`의 배치 스윕(B = 1–32, S = 4K–32K, N = 256, EOS 억제) 그림이다.
설계안은 [BATCH_PROFILING_PLAN.md](../BATCH_PROFILING_PLAN.md) §9(F1–F10)에 있다.
기록은 `dflash.batch_report.load()`로 읽는다. `python -m dflash.batch_report`가 출력하는
표와 같은 파일을 쓰고, `stale_*` 디렉터리는 읽지 않는다.

```bash
python visualization_batch/plot_bound.py        # F4, F7: memory-bound → compute-bound
python visualization_batch/plot_bottlenecks.py  # F1–F3, F5, F6, F8–F10

# 같은 그림을 다른 스윕에 대해 그린다. 파일 이름에 suffix가 붙는다.
python visualization_batch/plot_bound.py       --records record_batch_fla --suffix _fla
python visualization_batch/plot_bottlenecks.py --records record_batch_fla --suffix _fla
```

두 스윕이 있다.

| 디렉터리 | 내용 |
| --- | --- |
| `record_batch/` | 두 모델, GDN은 torch fallback(fla 없음). 2026-09-10 |
| `record_batch_fla/` | 9B만, fla 0.5.2 kernel. 2026-09-12. 출력 파일은 `_fla` 접미사 |

8B에는 GDN이 없어 fla와 무관하므로 다시 측정하지 않았다. 각 레코드는 `gdn_kernels`
필드에 실제로 바인딩된 구현과 fla 버전을 적는다. 그림의 각주 문구도 이 필드에서 온다.

| 파일 | 내용 |
| --- | --- |
| `_batch_style.py` | palette, 기록 로딩, analytic byte/FLOP 모델(verify, draft) |
| `batch_bound_<model>.{png,pdf}` | (a) roofline (b) arithmetic intensity vs B (c) ceiling 대비 달성률 vs B |
| `batch_throughput_<model>.{png,pdf}` | (a) decode tok/s (b) same-B speedup, iso-memory (c) τ (d) straggler tail (e) 출력 일치율 |
| `batch_step_<model>.{png,pdf}` | (a) step 시간 분해 (b) 토큰당 시간 분해 (c) step 비용 비율과 행-step당 토큰 |
| `batch_memory_<model>.{png,pdf}` | (a) 피크 시점 성분 분해 (b) throughput–피크 메모리 (c) DFlash phase별 피크 |
| `batch_verify_breakdown.{png,pdf}` | 고정 shape verify forward의 module별, kernel 카테고리별 시간(GPU idle 포함) |
| `batch_bound_metrics.csv` | 점마다 bytes, FLOPs, intensity, GB/s, TFLOP/s, ceiling 대비 비율, 판정 |
| `batch_metrics.csv` | 점마다 throughput, step 분해, 피크, τ, speedup |
| `batch_verify_breakdown.csv` | verify breakdown 표 |

## 기준선

* **Roofline의 지붕**은 이 카드에서 실측한 값이다(`record_batch/ceiling.json`). copy
  720 GB/s, bf16 GEMM 132 TFLOP/s이고 ridge는 183 FLOP/B다. 점선은 data sheet 값(768 GB/s,
  154.8 TFLOP/s)이다. 회색 파선은 M 토큰 GEMM(4096×12288)의 실측치로, 작은 배치에서 GEMM이
  실제로 도달하는 성능이다.
* **Intensity vs B**에는 ridge와 두 점근선이 있다. weights-only(≈ B·q FLOP/B)는 B와 함께
  오르고, KV-only(q·H_q/H_kv, verify는 64, baseline은 4)는 B와 무관하다. verify는 이 두 선
  사이에 있다. S가 길수록 KV-only 쪽으로 붙는다.
* **달성률**에는 §5.4의 판정 규칙을 선으로 그렸다. copy BW의 70% 이상이면 HBM-bound, GEMM
  peak의 60% 이상이면 compute-bound다.
* **Step 비용 비율 (c)**: `makespan speedup = (끝난 행을 포함한 행-step당 토큰) / (DF step
  / BL step)`이 정확히 성립한다. 따라서 자홍색 선이 점선 위로 올라가는 B가 break-even이다.
  파선 τ와 점선 사이의 간격은 static batch의 꼬리 비용이다.

## 읽을 때 주의

1. **byte와 FLOP은 analytic 값이다.** 옮겨야만 하는 양만 센다. 즉 weight 1회, KV 1회, GDN
   state in/out 1회다. 점이 지붕에서 멀리 떨어져 있다면 계산을 잘못한 것이 아니라 kernel이나
   launch 비효율이다. 어느 쪽인지는 `batch_verify_breakdown`의 GPU idle(빗금)과 attention
   막대로 확인한다.
2. **9B의 GDN kernel을 확인하고 읽어라.** `record_batch/`는 torch fallback,
   `record_batch_fla/`는 fla 0.5.2다. fla는 baseline을 더 많이 빠르게 만들어서 큰 B의
   speedup은 오히려 낮아졌다. 두 스윕의 점을 섞어서 비교하면 안 된다.
3. **9B baseline의 accept/commit에는 불필요한 `torch.where`가 들어 있다.** B=32에서 12 ms다.
   큰 B의 9B speedup은 그만큼 DFlash에 유리하게 나와 있다.
4. **두 모드 모두 drafter weight를 load한 상태로 측정했다.** 같은 프로세스가 두 모드를 모두
   돌리기 때문이다. 순수 baseline 프로세스라면 피크가 1.95 GiB(8B) 또는 2.41 GiB(9B) 낮다.
   이 격자에서 max B는 바뀌지 않는다.
5. **S마다 프롬프트의 task 구성이 다르다.** S에 따른 τ 변화에는 길이 효과와 task 효과가
   섞여 있다.
