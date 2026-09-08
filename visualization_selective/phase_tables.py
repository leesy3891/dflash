"""Markdown tables for PROFILING2.md, from the phase-instrumented records.

Reads ``record_selective/``, keeps only the records that carry per-phase memory
(``summary.dflash.phase_memory``), and prints the tables PROFILING2.md quotes.
Older records in the same directory are ignored rather than mixed in.

    python visualization_selective/phase_tables.py [record_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["qwen3-8b", "qwen3.5-9b"]
LENGTHS = [4096, 8192, 16384, 32768, 65536]


def load(record_dir: Path) -> dict:
    """The newest phase-instrumented record for each (model, context length)."""
    best: dict[tuple[str, int], tuple[str, dict]] = {}
    for path in sorted(record_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if not payload.get("summary", {}).get("dflash", {}).get("phase_memory"):
            continue
        key = (payload["model_name"], payload["context_length"])
        if key not in best or path.name > best[key][0]:
            best[key] = (path.name, payload)
    return {key: value[1] for key, value in best.items()}


def table(header: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(header) + " |"
    rule = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def each(records: dict):
    for model in ORDER:
        for length in LENGTHS:
            payload = records.get((model, length))
            if payload is not None:
                yield model, length, payload


def headline(records: dict) -> str:
    rows = []
    for model, length, payload in each(records):
        dflash = payload["summary"]["dflash"]
        base = payload["summary"].get("baseline")
        rows.append([
            model, f"{length // 1024}k", str(payload["num_devices"]),
            f"{dflash['mean_acceptance_length']:.2f}",
            f"{dflash['peak_memory_gb']:.2f}",
            f"{base['peak_memory_gb']:.2f}" if base else "-",
            f"{dflash['aggregate_time_per_output_token_s'] * 1000:.1f}",
            f"{base['aggregate_time_per_output_token_s'] * 1000:.1f}" if base else "-",
            f"{payload['decoding_speedup']:.2f}x" if base else "-",
        ])
    return table(
        ["model", "ctx", "GPUs", "accept", "peak GB", "base peak GB",
         "tpot ms", "base tpot ms", "speedup"],
        rows,
    )


def peak_phases(records: dict) -> str:
    rows = []
    for model, length, payload in each(records):
        dflash = payload["summary"]["dflash"]
        name = dflash["peak_phase"]
        phase = dflash["phase_memory"][name]
        parts = phase["peak_components_gb"]
        rows.append([
            model, f"{length // 1024}k", name,
            f"{phase['max_interval_peak_gb']:.2f}",
            f"{parts.get('target_weight_bytes', 0):.2f}",
            f"{parts.get('target_kv_bytes', 0):.2f}",
            f"{parts.get('draft_weight_bytes', 0):.2f}",
            f"{parts.get('draft_kv_bytes', 0):.2f}",
            f"{parts.get('selected_hidden_bytes', 0):.2f}",
            f"{parts.get('context_feature_bytes', 0):.2f}",
            f"{phase['peak_unattributed_gb']:.2f}",
        ])
    return table(
        ["model", "ctx", "peak phase", "peak GB", "tgt W", "tgt KV",
         "drf W", "drf KV", "sel hid", "ctx feat", "unattributed"],
        rows,
    )


def phase_profile(records: dict, model: str) -> str:
    """Every phase's largest occurrence, for one model, across the lengths."""
    lengths = [length for m, length, _ in each(records) if m == model]
    names: list[str] = []
    for length in lengths:
        for name in records[(model, length)]["summary"]["dflash"]["phase_memory"]:
            if name not in names:
                names.append(name)
    rows = []
    for name in names:
        row = [name]
        for length in lengths:
            phases = records[(model, length)]["summary"]["dflash"]["phase_memory"]
            row.append(
                f"{phases[name]['max_interval_peak_gb']:.2f}" if name in phases else "-"
            )
        rows.append(row)
    return table(["phase (peak GB)"] + [f"{n // 1024}k" for n in lengths], rows)


def first_draft(records: dict) -> str:
    rows = []
    for model, length, payload in each(records):
        d = payload["summary"]["dflash"]
        stages = d.get("first_draft_stage_s") or {}
        rows.append([
            model, f"{length // 1024}k",
            f"{(d['mean_first_draft_forward_s'] or 0) * 1000:.0f}",
            f"{(d['mean_first_draft_fraction_of_decode'] or 0) * 100:.1f}%",
            f"{(d['mean_steady_draft_forward_s'] or 0) * 1000:.2f}",
            f"{(d['mean_first_draft_forward_s'] or 0) / max(d['mean_steady_draft_forward_s'] or 1e-9, 1e-9):.0f}x",
            f"{d['max_first_draft_transient_gb']:.2f}",
            f"{d['max_first_draft_cache_gb']:.2f}",
            *[f"{stages.get(key, 0) * 1000:.1f}" for key in
              ("context_projection", "context_kv_projection", "cache_update",
               "attention", "output_head")],
        ])
    return table(
        ["model", "ctx", "first ms", "of decode", "steady ms", "first/steady",
         "first borrowed GB", "draft KV GB",
         "fc+norm ms", "ctx K/V ms", "KV append ms", "attn ms", "head ms"],
        rows,
    )


def steady_draft(records: dict) -> str:
    rows = []
    for model, length, payload in each(records):
        d = payload["summary"]["dflash"]
        rows.append([
            model, f"{length // 1024}k",
            f"{d['mean_steady_draft_calls']:.1f}",
            f"{(d['mean_steady_draft_forward_s'] or 0) * 1000:.2f}",
            f"{(d['mean_steady_attention_s'] or 0) * 1000:.2f}",
            f"{(d['mean_steady_cache_update_s'] or 0) * 1000:.2f}",
            f"{(d['mean_steady_context_kv_projection_s'] or 0) * 1000:.2f}",
            f"{(d['mean_steady_context_projection_s'] or 0) * 1000:.2f}",
            f"{(d['mean_steady_output_head_s'] or 0) * 1000:.2f}",
            f"{d['max_steady_draft_cache_gb']:.2f}",
            f"{d['max_steady_draft_transient_gb']:.2f}",
        ])
    return table(
        ["model", "ctx", "calls", "forward ms", "attn ms", "KV append ms",
         "ctx K/V ms", "fc+norm ms", "head ms", "draft KV GB", "borrowed GB"],
        rows,
    )


def overhead(records: dict) -> str:
    rows = []
    for model, length, payload in each(records):
        split = payload["summary"]["dflash"]["overhead_split"]
        base = payload["summary"].get("baseline") or {}
        dflash_total = (
            split["dflash_draft_weight_gb"]
            + split["dflash_draft_kv_gb"]
            + split["dflash_selected_hidden_gb"]
            + split["dflash_context_feature_gb"]
        )
        rows.append([
            model, f"{length // 1024}k",
            f"{split['dflash_draft_weight_gb']:.2f}",
            f"{split['dflash_draft_kv_gb']:.2f}",
            f"{split['dflash_selected_hidden_gb']:.2f}",
            f"{split['dflash_context_feature_gb']:.2f}",
            f"{dflash_total:.2f}",
            f"{split['target_kv_gb']:.2f}",
            f"{split['target_prefill_transient_gb']:.2f}",
            f"{base.get('max_target_prefill_transient_gb', 0):.2f}",
        ])
    return table(
        ["model", "ctx", "draft W", "draft KV", "sel hidden", "ctx feature",
         "DFlash total", "target KV", "target prefill act",
         "same, baseline"],
        rows,
    )


def main() -> None:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "record_selective")
    records = load(directory)
    if not records:
        raise SystemExit(f"no phase-instrumented records in {directory}")
    print("## Headline\n")
    print(headline(records))
    print("\n## Peak phase and its component split\n")
    print(peak_phases(records))
    for model in ORDER:
        if any(m == model for m, _, _ in each(records)):
            print(f"\n## Phase profile: {model}\n")
            print(phase_profile(records, model))
    print("\n## First draft call\n")
    print(first_draft(records))
    print("\n## Steady-state draft calls\n")
    print(steady_draft(records))
    print("\n## DFlash-specific vs target-architecture memory\n")
    print(overhead(records))


if __name__ == "__main__":
    main()
