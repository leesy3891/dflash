"""Aggregation and on-disk records for context-length benchmarks."""

from __future__ import annotations

import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

_GB = float(1 << 30)


def sample_metrics(stats, *, drafter: bool = True) -> dict:
    """Flatten one ``dflash_generate(..., return_stats=True)`` result.

    The baseline runs at ``block_size=1`` and never calls the drafter, so with
    ``drafter=False`` the drafter-specific fields are left out entirely rather
    than recorded as zeros.
    """
    metrics = {
        "num_input_tokens": stats.num_input_tokens,
        "num_output_tokens": stats.num_output_tokens,
        "time_to_first_token_s": stats.time_to_first_token,
        "total_latency_s": stats.total_latency,
        "decode_latency_s": stats.decode_latency,
        "time_per_output_token_s": stats.time_per_output_token,
        "num_decode_steps": stats.num_verify_steps,
        "peak_memory_bytes": stats.peak_memory_bytes,
        "peak_memory_reserved_bytes": stats.peak_memory_reserved_bytes,
        "target_cache_bytes": stats.target_cache_bytes,
        "target_forward_s": stats.target_forward_s,
    }
    if not drafter:
        return metrics
    return {
        **metrics,
        "num_accepted_tokens": stats.num_accepted_tokens,
        "num_proposed_tokens": stats.num_proposed_tokens,
        "num_verify_steps": stats.num_verify_steps,
        "num_draft_calls": stats.num_draft_calls,
        "num_full_gamma_proposals": stats.num_full_gamma_proposals,
        "gamma": stats.gamma,
        "draft_cache_bytes": stats.draft_cache_bytes,
        "draft_activation_bytes": stats.draft_activation_bytes,
        "target_hidden_states_bytes": stats.target_hidden_states_bytes,
        "context_feature_bytes": stats.context_feature_bytes,
        "draft_forward_s": stats.draft_forward_s,
        "context_feature_s": stats.context_feature_s,
        "accepted_lengths": stats.accepted_lengths,
        "acceptance_lengths": stats.acceptance_lengths,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _maximum(values: list, default=0):
    present = [v for v in values if v is not None]
    return max(present) if present else default


def _total_or_none(runs: list[dict], key: str):
    """Sum a per-sample timing, or None when the timing was not collected."""
    values = [r.get(key) for r in runs]
    return sum(values) if all(v is not None for v in values) else None


def summarize(
    runs: list[dict],
    block_size: int,
    draft_weight_bytes: int,
    *,
    drafter: bool = True,
) -> dict:
    """Aggregate per-sample metrics for one decoding configuration.

    With ``drafter=False`` only the model-agnostic metrics are produced, which
    is what the ``block_size=1`` baseline can meaningfully report.
    """
    total_output = sum(r["num_output_tokens"] for r in runs)
    total_decode = sum(r["decode_latency_s"] for r in runs)
    ttfts = [r["time_to_first_token_s"] for r in runs]

    summary = {
        "total_input_tokens": sum(r["num_input_tokens"] for r in runs),
        "mean_input_tokens": statistics.mean(r["num_input_tokens"] for r in runs),
        "total_output_tokens": total_output,
        "mean_output_tokens": statistics.mean(r["num_output_tokens"] for r in runs),
        # Latency
        "total_latency_s": sum(r["total_latency_s"] for r in runs),
        "mean_latency_s": statistics.mean(r["total_latency_s"] for r in runs),
        "mean_ttft_s": statistics.mean(ttfts),
        "p50_ttft_s": _percentile(ttfts, 0.50),
        "p95_ttft_s": _percentile(ttfts, 0.95),
        "mean_time_per_output_token_s": statistics.mean(
            r["time_per_output_token_s"] for r in runs
        ),
        "aggregate_time_per_output_token_s": total_decode / total_output,
        "decode_throughput_tok_s": total_output / total_decode,
        # Memory
        "peak_memory_gb": _maximum([r["peak_memory_bytes"] for r in runs]) / _GB,
        # Allocated is what the tensors occupy; reserved is what the caching
        # allocator holds. nvidia-smi shows reserved plus the CUDA context, so
        # reserved is the figure to reconcile against it.
        "peak_memory_reserved_gb": _maximum(
            [r["peak_memory_reserved_bytes"] for r in runs]
        )
        / _GB,
        "max_target_cache_gb": _maximum([r["target_cache_bytes"] for r in runs]) / _GB,
    }
    if not drafter:
        summary["mean_decode_steps"] = statistics.mean(
            r["num_decode_steps"] for r in runs
        )
        return summary

    total_proposed = sum(r["num_proposed_tokens"] for r in runs)
    total_accepted = sum(r["num_accepted_tokens"] for r in runs)
    total_steps = sum(r["num_verify_steps"] for r in runs)
    produced = [n for r in runs for n in r["acceptance_lengths"]]
    max_draft_cache = _maximum([r["draft_cache_bytes"] for r in runs])
    max_hidden_states = _maximum([r["target_hidden_states_bytes"] for r in runs])
    max_context_feature = _maximum([r["context_feature_bytes"] for r in runs])
    max_draft_activation = (
        _maximum([r["draft_activation_bytes"] for r in runs])
        if any(r["draft_activation_bytes"] is not None for r in runs)
        else None
    )
    draft_forward_s = _total_or_none(runs, "draft_forward_s")
    context_feature_s = _total_or_none(runs, "context_feature_s") or 0.0
    target_forward_s = _total_or_none(runs, "target_forward_s")
    summary.update(
        {
            # Acceptance
            "acceptance_rate": total_accepted / total_proposed if total_proposed else None,
            "mean_acceptance_length": total_steps and sum(produced) / total_steps,
            "acceptance_length_histogram": [
                produced.count(n) / len(produced) for n in range(block_size + 1)
            ]
            if produced
            else [],
            "total_accepted_tokens": total_accepted,
            "total_proposed_tokens": total_proposed,
            # Drafter activity
            "gamma": block_size - 1,
            "total_verify_steps": total_steps,
            "total_draft_calls": sum(r["num_draft_calls"] for r in runs),
            "total_full_gamma_proposals": sum(
                r["num_full_gamma_proposals"] for r in runs
            ),
            "mean_full_gamma_proposals_per_sample": statistics.mean(
                r["num_full_gamma_proposals"] for r in runs
            ),
            # Drafter memory
            "draft_weight_gb": draft_weight_bytes / _GB,
            "max_draft_cache_gb": max_draft_cache / _GB,
            "max_draft_activation_gb": (
                max_draft_activation / _GB if max_draft_activation is not None else None
            ),
            # Target-KV injection: DFlash feeds the target's per-layer context
            # feature into every draft layer, which both forces the target to
            # emit all hidden states and materialises the concatenated feature.
            # The baseline pays neither, so both are drafter overhead.
            "max_target_hidden_states_gb": max_hidden_states / _GB,
            "max_context_feature_gb": max_context_feature / _GB,
            "draft_overhead_gb": (
                draft_weight_bytes
                + max_draft_cache
                + max_hidden_states
                + max_context_feature
                + (max_draft_activation or 0)
            )
            / _GB,
            # Drafter latency
            "draft_forward_s": draft_forward_s,
            "context_feature_s": context_feature_s,
            "drafter_latency_s": (
                None
                if draft_forward_s is None
                else draft_forward_s + context_feature_s
            ),
            "target_forward_s": target_forward_s,
            "drafter_share_of_decode": (
                None
                if draft_forward_s is None
                else (draft_forward_s + context_feature_s) / total_decode
            ),
            "target_share_of_decode": (
                None if target_forward_s is None else target_forward_s / total_decode
            ),
        }
    )
    return summary


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write(record_dir: str, model_name: str, context_length: int, payload: dict) -> Path:
    """Write ``payload`` to ``<record_dir>/<model>_<context>_<date>.json``."""
    directory = Path(record_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"{model_name}_{context_length}_{stamp}.json"
    payload = {"git_commit": _git_commit(), **payload}
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_summary(summaries: dict[str, dict], block_size: int) -> None:
    dflash = summaries["dflash"]
    baseline = summaries.get("baseline")

    def row(label: str, value: str) -> None:
        print(f"  {label:<40}{value}")

    def common(summary: dict) -> None:
        row("Input tokens (total / mean)", f"{summary['total_input_tokens']} / {summary['mean_input_tokens']:.1f}")
        row("Output tokens (total / mean)", f"{summary['total_output_tokens']} / {summary['mean_output_tokens']:.1f}")
        row("Total latency", f"{summary['total_latency_s']:.1f}s")
        row("TTFT mean / p50 / p95", f"{summary['mean_ttft_s']:.3f}s / {summary['p50_ttft_s']:.3f}s / {summary['p95_ttft_s']:.3f}s")
        row("Per-decode-token latency", f"{summary['aggregate_time_per_output_token_s'] * 1000:.2f}ms")
        row("Decode throughput", f"{summary['decode_throughput_tok_s']:.2f} tok/s")
        row(
            "Peak memory (allocated / reserved)",
            f"{summary['peak_memory_gb']:.2f} / {summary['peak_memory_reserved_gb']:.2f} GB",
        )
        row("Target KV cache (max)", f"{summary['max_target_cache_gb']:.2f} GB")

    print(f"\n{'=' * 64}")
    print("DFlash (block_size=%d)" % block_size)
    common(dflash)
    print("  -- acceptance")
    row("Acceptance rate (accepted/proposed)", f"{dflash['acceptance_rate']:.4f}")
    row("Mean acceptance length", f"{dflash['mean_acceptance_length']:.2f}")
    row(
        "Acceptance histogram",
        str([f"{x * 100:.1f}%" for x in dflash["acceptance_length_histogram"]]),
    )
    print("  -- drafter activity")
    row("Gamma (tokens per proposal)", str(dflash["gamma"]))
    row("Full-gamma proposals", str(dflash["total_full_gamma_proposals"]))
    row("Draft calls / verify steps", f"{dflash['total_draft_calls']} / {dflash['total_verify_steps']}")
    print("  -- drafter memory overhead")
    row("Draft weights", f"{dflash['draft_weight_gb']:.2f} GB")
    row("Draft KV cache (max)", f"{dflash['max_draft_cache_gb']:.2f} GB")
    row("Target hidden states (max)", f"{dflash['max_target_hidden_states_gb']:.2f} GB")
    row("Injected context feature (max)", f"{dflash['max_context_feature_gb']:.2f} GB")
    if dflash["max_draft_activation_gb"] is not None:
        row("Draft activation peak (max)", f"{dflash['max_draft_activation_gb']:.2f} GB")
    row("Total drafter overhead", f"{dflash['draft_overhead_gb']:.2f} GB")
    if dflash["drafter_latency_s"] is not None:
        print("  -- drafter latency")
        row("Draft forward", f"{dflash['draft_forward_s']:.1f}s")
        row("Context-feature build", f"{dflash['context_feature_s']:.1f}s")
        row(
            "Drafter total (share of decode)",
            f"{dflash['drafter_latency_s']:.1f}s ({dflash['drafter_share_of_decode'] * 100:.1f}%)",
        )
        row(
            "Target verify (share of decode)",
            f"{dflash['target_forward_s']:.1f}s ({dflash['target_share_of_decode'] * 100:.1f}%)",
        )

    if baseline is None:
        print(f"{'=' * 64}")
        return

    print("\nBaseline (block_size=1, drafter unused)")
    common(baseline)
    row("Decode steps (mean)", f"{baseline['mean_decode_steps']:.1f}")

    print("\nDFlash vs baseline")
    row(
        "Decoding speedup",
        f"{baseline['aggregate_time_per_output_token_s'] / dflash['aggregate_time_per_output_token_s']:.2f}x",
    )
    row(
        "Peak allocated delta",
        f"{dflash['peak_memory_gb'] - baseline['peak_memory_gb']:.2f} GB",
    )
    row("Drafter memory overhead", f"{dflash['draft_overhead_gb']:.2f} GB")
    print(f"{'=' * 64}")
