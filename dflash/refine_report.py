"""Markdown tables for the local-refinement sweep.

    python -m dflash.refine_report record_refine > tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_MIB = float(1 << 20)
_GIB = float(1 << 30)


def load(record_dir: str) -> list[dict]:
    """The newest record per (model, context length) in each directory.

    Several directories may be given, comma-separated; records are kept apart
    per directory so parallel sweeps over the same context stay visible.
    """
    latest: dict = {}
    paths = [
        path
        for directory in record_dir.split(",")
        for path in sorted(Path(directory).glob("*.json"))
    ]
    for path in paths:
        record = json.loads(path.read_text())
        record["_path"] = path.name
        latest[(record["model_name"], record["context_length"], str(path.parent))] = record
    return [latest[key] for key in sorted(latest)]


def _ctx(record) -> str:
    return f"{record['context_length'] // 1024}k"


def _configs(record) -> list[str]:
    names = [n for n in record["summary"] if n not in ("baseline", "dflash")]

    def key(name):
        config = record["summary"][name].get("local_refine") or {}
        window = config.get("window", "0")
        return (
            float(config.get("alpha", 1.0)),
            1e9 if window == "full" else int(window),
        )

    return ["dflash"] + sorted(names, key=key)


def _label(name: str) -> str:
    if name == "dflash":
        return "DFlash"
    parts = name.split("_")
    label = f"+refine w={parts[1][1:]}"
    alpha = next((part[1:] for part in parts[3:] if part.startswith("a")), None)
    return label + ("" if alpha is None else f" a={alpha}")


def _fmt(value, spec, scale=1.0, none="n/a"):
    return none if value is None else format(value * scale, spec)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def headline(records) -> str:
    rows = []
    for record in records:
        summary = record["summary"]
        base = summary["baseline"]
        base_tpot = base["aggregate_time_per_output_token_s"]
        rows.append([
            _ctx(record), "baseline", "1.00", "-", "-",
            _fmt(base_tpot, ".2f", 1000),
            _fmt(base_tpot * base["total_output_tokens"], ".1f"),
            "1.00x", "-", "-", _fmt(base["peak_memory_gb"], ".3f"),
        ])
        for name in _configs(record):
            s = summary[name]
            tpot = s["aggregate_time_per_output_token_s"]
            rows.append([
                _ctx(record),
                _label(name),
                _fmt(s["mean_acceptance_length"], ".3f"),
                f"{s['total_accepted_tokens']} / {s['total_proposed_tokens']}",
                _fmt(s["acceptance_rate"], ".4f"),
                _fmt(tpot, ".2f", 1000),
                _fmt(tpot * s["total_output_tokens"], ".1f"),
                f"{base_tpot / tpot:.3f}x",
                _fmt(s["mean_steady_draft_forward_s"], ".2f", 1000),
                _fmt(s.get("refine_share_of_decode"), ".2f", 100, "-"),
                _fmt(s["peak_memory_gb"], ".3f"),
            ])
    return _table(
        ["ctx", "config", "accept len", "accepted / proposed", "accept rate",
         "TPOT ms", "decode s", "speedup", "steady draft fwd ms",
         "refine % of decode", "peak GiB"],
        rows,
    )


def stages(records) -> str:
    rows = []
    for record in records:
        summary = record["summary"]
        steady = summary["dflash"]["mean_steady_draft_forward_s"]
        for name in _configs(record):
            if name == "dflash":
                continue
            s = summary[name]
            total = s["mean_refine_total_ms_per_draft_call"]
            rows.append([
                _ctx(record), _label(name), str(s["total_refine_calls"]),
                _fmt(s["mean_refine_soft_embedding_ms"], ".3f"),
                _fmt(s["mean_refine_kv_projection_ms"], ".3f"),
                _fmt(s["mean_refine_attention_ms"], ".3f"),
                _fmt(s["mean_refine_rerank_ms"], ".3f"),
                _fmt(total, ".3f"),
                _fmt(None if steady is None else total / (steady * 1000), ".1f", 100),
                _fmt(s["refine_share_of_decode"], ".2f", 100),
            ])
    return _table(
        ["ctx", "config", "calls", "soft embedding ms", "K/V projection ms",
         "attention ms", "candidate rerank ms", "total ms / draft call",
         "% of DFlash steady draft fwd", "% of decode"],
        rows,
    )


def memory(records) -> str:
    rows = []
    for record in records:
        summary = record["summary"]
        base_peak = summary["baseline"]["peak_memory_gb"]
        df = summary["dflash"]
        for name in _configs(record):
            s = summary[name]
            phase = (s.get("phase_memory") or {}).get("decode: local refine") or {}
            rows.append([
                _ctx(record), _label(name),
                _fmt(s["peak_memory_gb"], ".4f"),
                _fmt(s["peak_memory_gb"] - base_peak, "+.4f"),
                _fmt(s["peak_memory_gb"] - df["peak_memory_gb"], "+.4f"),
                str(s.get("peak_phase")),
                _fmt(s["max_steady_draft_transient_gb"], ".2f", 1024),
                _fmt(s.get("refine_peak_transient_bytes"), ".2f", 1 / _MIB, "-"),
                _fmt(phase.get("max_interval_peak_gb"), ".3f", 1, "-"),
            ])
    return _table(
        ["ctx", "config", "peak GiB", "vs baseline GiB", "vs DFlash GiB",
         "peak phase", "steady draft transient MiB",
         "refine_peak_transient MiB", "refine phase peak GiB"],
        rows,
    )


def acceptance(records) -> str:
    rows = []
    for record in records:
        summary = record["summary"]
        df = summary["dflash"]
        for name in _configs(record):
            if name == "dflash":
                continue
            s = summary[name]
            rows.append([
                _ctx(record), _label(name),
                f"{df['mean_acceptance_length']:.3f} -> {s['mean_acceptance_length']:.3f} "
                f"({s['mean_acceptance_length'] - df['mean_acceptance_length']:+.3f})",
                f"{df['total_accepted_tokens']} -> {s['total_accepted_tokens']}",
                f"{df['total_verify_steps']} -> {s['total_verify_steps']}",
                f"{s['refine_changed_proposals']} ({s['refine_changed_fraction'] * 100:.2f}%)",
                str(s["refine_changed_verified_positions"]),
                str(s["refine_changed_helped"]),
                str(s["refine_changed_hurt"]),
            ])
    return _table(
        ["ctx", "config", "mean accept len (DFlash -> refine)",
         "accepted tokens", "verify steps", "proposals changed by rerank",
         "changed at verified positions", "helped", "hurt"],
        rows,
    )


def forward_counts(records) -> str:
    rows = []
    for record in records:
        summary = record["summary"]
        for name in _configs(record):
            s = summary[name]
            calls = s.get("module_calls_by_phase") or {}
            drafter = calls.get("draft_model_forward", {})
            head = calls.get("output_head_forward", {})
            rows.append([
                _ctx(record), _label(name), str(s["total_draft_calls"]),
                str(sum(drafter.values())),
                str(head.get("decode: draft logits", 0)),
                str(head.get("decode: target verify", 0)),
                str(drafter.get("decode: local refine", 0)),
                str(head.get("decode: local refine", 0)),
                ", ".join(sorted(set(drafter) | set(head))),
            ])
    return _table(
        ["ctx", "config", "draft calls", "drafter forwards (all phases)",
         "full-vocab head: draft logits", "full-vocab head: target verify",
         "drafter fwd in refine", "full-vocab head in refine", "phases seen"],
        rows,
    )


def exactness(records) -> str:
    rows = []
    for record in records:
        samples = record["samples"]
        for name in _configs(record):
            same_df = sum(
                s[name]["output_token_ids"] == s["dflash"]["output_token_ids"]
                for s in samples
            )
            same_base = sum(
                s[name]["output_token_ids"] == s["baseline"]["output_token_ids"]
                for s in samples
            )
            rows.append([
                _ctx(record), _label(name),
                f"{same_df}/{len(samples)}", f"{same_base}/{len(samples)}",
                str(record["summary"][name]["total_output_tokens"]),
            ])
    return _table(
        ["ctx", "config", "outputs == DFlash", "outputs == baseline",
         "output tokens"],
        rows,
    )


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    records = load(argv[0] if argv else "record_refine")
    for title, build in (
        ("Headline", headline),
        ("Refinement stage latency", stages),
        ("Memory", memory),
        ("Acceptance", acceptance),
        ("Forward counts", forward_counts),
        ("Exactness", exactness),
    ):
        print(f"### {title}\n")
        print(build(records))
        print()
    print("Records: " + ", ".join(r["_path"] for r in records))


if __name__ == "__main__":
    main()
