"""Tables from the batch sweep's records (no plots).

    python -m dflash.batch_report [record_batch]           # print the tables
    python -m dflash.batch_report --ceiling [record_batch]  # measure this GPU's
                                                            # copy / GEMM ceilings first

Byte and FLOP counts are analytic (BATCH_PROFILING_PLAN.md §5.4): the target's
executed weights (its parameters less the embedding table, which is gathered,
not streamed), plus the KV each verify reads at the mean context length. Divided
by the CUDA-event verify time they give an achieved bandwidth, which is compared
with a measured copy ceiling rather than the data-sheet figure.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_GB = float(1 << 30)
EMBED_BYTES = {"qwen3-8b": 151936 * 4096 * 2, "qwen3.5-9b": 248320 * 4096 * 2}


def load(directory: Path) -> dict:
    """Latest record per (model, context, batch)."""
    latest = {}
    for path in sorted(directory.glob("*_b*_*.json")):
        match = re.match(r"(.+)_(\d+)_b(\d+)_(\d{8}-\d{6})\.json", path.name)
        if not match:
            continue
        key = (match.group(1), int(match.group(2)), int(match.group(3)))
        latest[key] = path
    return {key: json.loads(path.read_text()) for key, path in latest.items()}


def measure_ceiling(path: Path) -> dict:
    import torch

    torch.cuda.synchronize()
    src = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    copy_gbps = 20 * 2 * src.numel() / (start.elapsed_time(end) / 1000) / 1e9
    gemm = {}
    weight = torch.randn(12288, 4096, dtype=torch.bfloat16, device="cuda")
    for m in (16, 64, 128, 256, 512, 1024, 4096):
        x = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            x @ weight.T
        start.record()
        for _ in range(50):
            x @ weight.T
        end.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(end) / 1000 / 50
        gemm[m] = {"tflops": 2 * m * 4096 * 12288 / seconds / 1e12,
                   "gbps": weight.numel() * 2 / seconds / 1e9}
    result = {"device": torch.cuda.get_device_name(0), "copy_gbps": copy_gbps, "gemm": gemm}
    path.write_text(json.dumps(result, indent=1))
    return result


def _fmt(value, spec=".2f", none="—"):
    return none if value is None else format(value, spec)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="record_batch")
    parser.add_argument("--ceiling", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    ceiling_path = directory / "ceiling.json"
    if args.ceiling:
        print(json.dumps(measure_ceiling(ceiling_path), indent=1))
    ceiling = json.loads(ceiling_path.read_text()) if ceiling_path.exists() else None
    records = load(directory)
    by_model = defaultdict(list)
    for key in sorted(records):
        by_model[key[0]].append(key)

    for model, keys in by_model.items():
        print(f"\n## {model}\n")
        print("### Throughput and speedup (decode tok/s, all rows; speedup = DFlash / baseline at the same B)\n")
        print("| S | B | BL tok/s | DF tok/s | speedup | speedup full-occ | vs BL B=1 | BL scale eff | DF scale eff | τ | τ pre-EOS | e2e speedup | status |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        b1 = {}
        for key in keys:
            rec = records[key]
            s = rec["summary"]
            bl, df = s.get("baseline"), s.get("dflash")
            if key[2] == 1:
                b1[key[1]] = (bl, df)
            base1 = b1.get(key[1], (None, None))
            sp = rec.get("speedup") or {}
            def eff(cur, ref):
                if not cur or not ref:
                    return None
                return cur["decode_tok_s_makespan"] / (key[2] * ref["decode_tok_s_makespan"])
            print(
                f"| {key[1] // 1024}k | {key[2]} | {_fmt(bl and bl['decode_tok_s_makespan'], '.1f')} "
                f"| {_fmt(df and df['decode_tok_s_makespan'], '.1f')} | {_fmt(sp.get('same_b_makespan'))} "
                f"| {_fmt(sp.get('same_b_full_occupancy'))} "
                f"| {_fmt(df and base1[0] and df['decode_tok_s_makespan'] / base1[0]['decode_tok_s_makespan'])} "
                f"| {_fmt(eff(bl, base1[0]))} | {_fmt(eff(df, base1[1]))} "
                f"| {_fmt(df and df.get('mean_acceptance_length'))} | {_fmt(df and df.get('mean_acceptance_length_pre_eos'))} "
                f"| {_fmt(sp.get('same_b_e2e'))} | {rec['status']} |"
            )

        print("\n### Decode step (ms per step; GPU parts from CUDA events, cpu = wall − GPU)\n")
        print("| S | B | mode | wall | verify | accept/rollback | ctx | draft | cpu | tok/step | verify GB/s | verify TFLOP/s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in keys:
            rec = records[key]
            for mode in ("baseline", "dflash"):
                s = rec["summary"].get(mode)
                if not s:
                    continue
                t = s["step_time_s"]
                q = 1 if mode == "baseline" else rec["block_size"]
                weights = s["target_weight_gb"] * _GB - EMBED_BYTES.get(model, 0)
                kv_per_token = s["target_kv_alloc_gb"] * _GB / (key[2] * rec["batches"][mode][0]["capacity"])
                mean_len = key[1] + rec["fixed_output_tokens"] / 2
                read = weights + key[2] * mean_len * kv_per_token
                # Steps where some rows had finished still run B rows.
                gbps = read / t["verify"] / 1e9 if t.get("verify") else None
                tflops = 2 * (weights / 2) * key[2] * q / t["verify"] / 1e12 if t.get("verify") else None
                print(
                    f"| {key[1] // 1024}k | {key[2]} | {mode} | {s['mean_step_wall_s'] * 1e3:.1f} | "
                    + " | ".join(_fmt(t.get(p) and t[p] * 1e3, ".1f") for p in ("verify", "accept", "ctx", "draft", "cpu_and_sync"))
                    + f" | {_fmt(s['mean_tokens_per_step'], '.1f')} | {_fmt(gbps, '.0f')} | {_fmt(tflops, '.1f')} |"
                )

        print("\n### Memory at the peak (GB; components read at one instant inside the peak phase)\n")
        print("| S | B | mode | peak | peak phase | tgt W | tgt KV (alloc) | GDN state | drf W | drf KV | sel hidden | ctx feat | unattributed |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in keys:
            rec = records[key]
            for mode in ("baseline", "dflash"):
                s = rec["summary"].get(mode)
                if not s:
                    continue
                phase = s["peak_phase"]
                entry = s["phase_memory"][phase]
                c = entry["peak_components_gb"]
                print(
                    f"| {key[1] // 1024}k | {key[2]} | {mode} | {s['peak_memory_gb']:.2f} | {phase} "
                    f"| {c.get('target_weight_bytes', 0):.2f} | {c.get('target_kv_bytes', 0):.2f} "
                    f"| {c.get('gdn_state_bytes', 0):.2f} | {c.get('draft_weight_bytes', 0):.2f} "
                    f"| {c.get('draft_kv_bytes', 0):.2f} | {c.get('selected_hidden_bytes', 0):.2f} "
                    f"| {c.get('context_feature_bytes', 0):.2f} | {entry['peak_unattributed_gb']:.2f} |"
                )
            for mode, info in (rec.get("oom") or {}).items():
                print(f"| {key[1] // 1024}k | {key[2]} | {mode} | OOM | {info.get('phase')} | | | | | | | | |")

        print("\n### Correctness (greedy, fp/bf16): DFlash == baseline per row; each mode vs its own B=1 output\n")
        print("| S | B | DF == BL | BL == BL(B=1) | DF == DF(B=1) |")
        print("|---|---|---|---|---|")
        for key in keys:
            rec = records[key]
            ref = records.get((key[0], key[1], 1))
            rows = rec["rows"]
            def rate(pred):
                values = [pred(r) for r in rows]
                values = [v for v in values if v is not None]
                return f"{sum(values)}/{len(values)}" if values else "—"
            ref_rows = {r["prompt"]: r for r in ref["rows"]} if ref else {}
            def same_as_b1(mode):
                def pred(r):
                    other = ref_rows.get(r["prompt"], {}).get(mode)
                    return None if other is None or mode not in r else r[mode]["tokens"] == other["tokens"]
                return pred
            print(
                f"| {key[1] // 1024}k | {key[2]} | {rate(lambda r: r.get('dflash_matches_baseline'))} "
                f"| {rate(same_as_b1('baseline'))} | {rate(same_as_b1('dflash'))} |"
            )

        print("\n### Capacity: largest batch that ran, and the best throughput on one card\n")
        print("| S | mode | max B | peak at max B (GB) | first OOM (B, phase) | best tok/s (at B) |")
        print("|---|---|---|---|---|---|")
        by_context = defaultdict(list)
        for key in keys:
            by_context[key[1]].append(key)
        for context, ctx_keys in by_context.items():
            best = {}
            for mode in ("baseline", "dflash"):
                ok = [(k[2], records[k]["summary"][mode]) for k in ctx_keys if records[k]["summary"].get(mode)]
                oom = [(k[2], records[k]["oom"][mode]) for k in ctx_keys if mode in (records[k].get("oom") or {})]
                if not ok:
                    continue
                top_b, top = max(ok, key=lambda item: item[0])
                fast_b, fast = max(ok, key=lambda item: item[1]["decode_tok_s_makespan"])
                best[mode] = fast["decode_tok_s_makespan"]
                first_oom = f"B={oom[0][0]}, {oom[0][1].get('phase')}" if oom else "—"
                print(f"| {context // 1024}k | {mode} | {top_b} | {top['peak_memory_gb']:.2f} | {first_oom} "
                      f"| {fast['decode_tok_s_makespan']:.1f} (B={fast_b}) |")
            if len(best) == 2:
                print(f"| {context // 1024}k | **iso-memory speedup** | | | | "
                      f"**{best['dflash'] / best['baseline']:.2f}x** |")

    for path in sorted(directory.glob("verify_breakdown_*.json")):
        data = json.loads(path.read_text())
        print(f"\n## Verify forward breakdown — {data['model']} (ms per forward; module spans from CUDA events)\n")
        modules = sorted({m for r in data["rows"] for m in r["module_ms"]})
        kernels = sorted({m for r in data["rows"] for m in r["kernel_ms"]})
        print("| S | B | q | wall | " + " | ".join(modules) + " | kernels: " + " | ".join(kernels) + " |")
        print("|" + "---|" * (4 + len(modules) + len(kernels)))
        for r in data["rows"]:
            print(f"| {r['context'] // 1024}k | {r['batch']} | {r['q']} | {r['wall_ms']:.1f} | "
                  + " | ".join(_fmt(r["module_ms"].get(m), ".1f") for m in modules) + " | "
                  + " | ".join(_fmt(r["kernel_ms"].get(k), ".1f") for k in kernels) + " |")

    compares = sorted(directory.glob("compare_old_*.json"))
    if compares:
        print("\n## B=1 bridge: pre-batch dflash_generate vs the batched engine (EOS ignored)\n")
        print("| model | S | old BL ms | new BL ms | old DF ms | new DF ms | old speedup | new speedup "
              "| old DF==BL | new DF==BL | old BL==new BL | τ old / new |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for path in compares:
            s = json.loads(path.read_text())["summary"]
            t, same = s["tpot_ms"], s["identical_rows"]
            n = s["prompts"]
            print(f"| {s['model']} | {s['context_length'] // 1024}k | {t['old_baseline']:.1f} | {t['new_baseline']:.1f} "
                  f"| {t['old_dflash']:.1f} | {t['new_dflash']:.1f} | {s['speedup']['old']:.2f} | {s['speedup']['new']:.2f} "
                  f"| {same['old_dflash == old_baseline']}/{n} | {same['new_dflash == new_baseline']}/{n} "
                  f"| {same['old_baseline == new_baseline']}/{n} | {s['tau']['old_dflash']:.2f} / {s['tau']['new_dflash']:.2f} |")

    if ceiling:
        print(f"\nCeiling on {ceiling['device']}: copy {ceiling['copy_gbps']:.0f} GB/s; bf16 GEMM 4096x12288: "
              + ", ".join(f"M={m} {v['tflops']:.0f} TFLOP/s" for m, v in ceiling["gemm"].items()))


if __name__ == "__main__":
    main()
