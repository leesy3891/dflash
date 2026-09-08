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
        "peak_memory_sum_device_maxima_bytes": getattr(
            stats, "peak_memory_sum_device_maxima_bytes", None
        ),
        "peak_memory_per_device_bytes": getattr(
            stats, "peak_memory_per_device_bytes", None
        ),
        "peak_site": getattr(stats, "peak_site", None),
        "target_cache_bytes": stats.target_cache_bytes,
        "target_forward_s": stats.target_forward_s,
        # Phase-local memory. Recorded for the baseline too, so the target's own
        # prefill activation can be read off a run that has no drafter in it and
        # subtracted from DFlash's peak rather than guessed at.
        "phase_memory": getattr(stats, "phase_memory", None),
        "target_prefill_transient_bytes": getattr(
            stats, "target_prefill_transient_bytes", None
        ),
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
        # The drafter's own prefill: the first draft call of the request, which
        # projects the whole context feature and fills the draft KV cache.
        "first_draft_forward_s": stats.first_draft_forward_s,
        "first_draft_peak_memory_bytes": stats.first_draft_peak_memory_bytes,
        "first_draft_allocated_before_bytes": stats.first_draft_allocated_before_bytes,
        "first_draft_cache_bytes": stats.first_draft_cache_bytes,
        "first_draft_cache_pre_crop_bytes": stats.first_draft_cache_pre_crop_bytes,
        "first_draft_fraction_of_decode": stats.first_draft_fraction_of_decode,
        "first_draft_transient_bytes": stats.first_draft_transient_bytes,
        "first_draft_stage_s": stats.first_draft_stage_s,
        # Steady state: every draft call after the first.
        "steady_draft_calls": stats.steady_draft_calls,
        "mean_steady_draft_forward_s": stats.mean_steady_draft_forward_s,
        "mean_steady_attention_s": stats.mean_steady_attention_s,
        "mean_steady_cache_update_s": stats.mean_steady_cache_update_s,
        "mean_steady_context_kv_projection_s": stats.mean_steady_context_kv_projection_s,
        "mean_steady_context_projection_s": stats.mean_steady_context_projection_s,
        "mean_steady_output_head_s": stats.mean_steady_output_head_s,
        "steady_draft_cache_bytes": stats.steady_draft_cache_bytes,
        "steady_draft_peak_memory_bytes": stats.steady_draft_peak_memory_bytes,
        "steady_draft_transient_bytes": stats.steady_draft_transient_bytes,
        # Classification of what the drafter keeps resident.
        "draft_weight_bytes": stats.draft_weight_bytes,
        "prefill_context_feature_bytes": stats.prefill_context_feature_bytes,
        "steady_context_feature_bytes": stats.steady_context_feature_bytes,
        "dflash_persistent_resident_bytes": stats.dflash_persistent_resident_bytes,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _peak_site(runs: list[dict]) -> dict | None:
    """The site of the largest peak across samples, with its sample index."""
    best = None
    for index, run in enumerate(runs):
        site = run.get("peak_site")
        if site is None:
            continue
        if best is None or run["peak_memory_bytes"] > best[0]:
            best = (
                run["peak_memory_bytes"],
                {
                    **site,
                    "sample": index,
                    "peak_gb": run["peak_memory_bytes"] / _GB,
                    "per_device_gb": [
                        b / _GB for b in (run.get("peak_memory_per_device_bytes") or [])
                    ],
                },
            )
    return best[1] if best else None


def describe_peak_site(site: dict | None) -> str:
    """One line naming the operation, the step, and the layer that peaked.

    The peak of a run is one interval of ``PeakTracker``, and every interval
    carries the operation it ran plus whatever positional context that phase
    has: the decode token and drafter step in the decode loop, the prefill
    chunk during prefill, the decoder layer index when the layer pre-hooks are
    on. This renders whichever of those the site actually has, so a prefill
    peak reads as a layer and a decode peak reads as a step.
    """
    if not site:
        return "unknown"
    where = str(site.get("operation", "?"))
    parts = []
    if site.get("decode_token") is not None:
        parts.append(f"decode token {site['decode_token']}")
    if site.get("draft_step") is not None:
        parts.append(f"drafter step {site['draft_step']}")
    if site.get("layer") is not None and "layer" not in where:
        parts.append(f"target layer {site['layer']}")
    if site.get("chunk") is not None:
        parts.append(f"prefill chunk {site['chunk']}")
    if site.get("sample") is not None:
        parts.append(f"sample {site['sample']}")
    return where + (" @ " + ", ".join(parts) if parts else "")


def _peak_site_histogram(runs: list[dict]) -> dict:
    """How often each operation set a sample's peak.

    One sample's peak site can be an accident of where that prompt stopped;
    the distribution over samples says whether the operation is really the
    high-water mark of the configuration.
    """
    counts: dict[str, int] = {}
    for run in runs:
        site = run.get("peak_site")
        if site is None:
            continue
        key = str(site.get("operation", "?"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _budget(summary: dict, *, drafter: bool) -> dict:
    """Split the peak into the tensors that are accounted for, plus the rest.

    Every term is a max over samples of a quantity that is resident for the
    whole run, so the sum is an upper bound on the resident part rather than a
    simultaneous reading. What is left over -- ``transient_gb`` -- is not the
    activation at the peak: the maxima it subtracts are taken at different
    moments (the target KV is largest at the end of decode, the prefill's
    residual streams before the first token exists, the peak itself inside the
    target's last prefill layer), so the remainder mixes three instants and can
    even come out negative.

    Kept because every record in the earlier sweeps carries it and dropping it
    would make them incomparable. ``summary["phase_memory"]`` is the figure to
    read instead: each of its rows is one instant. See PhaseMemory.
    """
    terms = {
        "target_weight_gb": summary["target_weight_gb"],
        "target_cache_gb": summary["max_target_cache_gb"],
    }
    if not drafter:
        # The drafter is loaded on the same device for the whole benchmark, so
        # its weights sit in the baseline's peak as well even though the
        # baseline never calls it. Listing it separately keeps the baseline's
        # transient term comparable with DFlash's instead of absorbing 2 GB of
        # weights, and is why a DFlash-minus-baseline peak delta understates
        # the drafter's true cost.
        terms["draft_weight_resident_gb"] = summary["draft_weight_gb"]
    if drafter:
        terms.update(
            {
                "draft_weight_gb": summary["draft_weight_gb"],
                "draft_cache_gb": summary["max_draft_cache_gb"],
                "target_hidden_states_gb": summary["max_target_hidden_states_gb"],
                "context_feature_gb": summary["max_context_feature_gb"],
            }
        )
        if summary.get("max_draft_activation_gb") is not None:
            terms["draft_activation_gb"] = summary["max_draft_activation_gb"]
    resident = sum(terms.values())
    return {
        **terms,
        "resident_total_gb": resident,
        "transient_gb": summary["peak_memory_gb"] - resident,
        "peak_memory_gb": summary["peak_memory_gb"],
    }


def _phase_names(runs: list[dict]) -> list[str]:
    """Every phase seen, in the order the phases first ran."""
    names: list[str] = []
    for run in runs:
        for name in run.get("phase_memory") or {}:
            if name not in names:
                names.append(name)
    return names


def _phase_transient(record: dict) -> float:
    """A phase's borrowed bytes: its peak less what stood on both sides of it."""
    resident = max(
        record["allocated_before_bytes"], record["allocated_after_bytes"]
    )
    return max(record["interval_peak_bytes"] - resident, 0)


def aggregate_phases(runs: list[dict], target_weight_bytes: int = 0) -> dict:
    """Per phase, the largest occurrence across samples and the per-sample means.

    The peak of a phase carries the component split read inside it, so the
    entries here are simultaneous decompositions: target KV, draft KV, selected
    hidden states, injected context feature and draft weights, all measured at
    one instant, plus the allocation that reading does not name.
    """
    aggregate: dict = {}
    for name in _phase_names(runs):
        entries = [
            (index, run["phase_memory"][name])
            for index, run in enumerate(runs)
            if name in (run.get("phase_memory") or {})
        ]
        peaks = [
            (entry["peak"]["interval_peak_bytes"], index, entry)
            for index, entry in entries
            if entry.get("peak")
        ]
        if not peaks:
            continue
        peak_bytes, peak_sample, peak_entry = max(peaks, key=lambda item: item[0])
        record = peak_entry["peak"]
        components = dict(record.get("components") or {})
        components.pop("allocated_bytes", None)
        # Resident at every instant of every phase, and not something the probe
        # has to read: the target's parameters. Listed with the rest so the
        # split at the peak adds up to the peak.
        components["target_weight_bytes"] = target_weight_bytes
        aggregate[name] = {
            "mean_occurrences_per_sample": statistics.mean(
                entry["count"] for _, entry in entries
            ),
            "mean_allocated_before_gb": statistics.mean(
                entry["mean_allocated_before_bytes"] for _, entry in entries
            )
            / _GB,
            "mean_allocated_after_gb": statistics.mean(
                entry["mean_allocated_after_bytes"] for _, entry in entries
            )
            / _GB,
            "mean_interval_peak_gb": statistics.mean(
                entry["mean_interval_peak_bytes"] for _, entry in entries
            )
            / _GB,
            "max_interval_peak_gb": peak_bytes / _GB,
            "max_at_sample": peak_sample,
            "peak_interval_label": record.get("peak_interval_label"),
            "peak_allocated_before_gb": record["allocated_before_bytes"] / _GB,
            "peak_allocated_after_gb": record["allocated_after_bytes"] / _GB,
            "peak_transient_gb": _phase_transient(record) / _GB,
            "peak_components_gb": {
                key: value / _GB for key, value in components.items()
            },
            # What the reading at the peak does not name: attention
            # workspaces, logits, the target's own layer activations. This is
            # one instant minus the components live at that same instant, so
            # unlike transient_gb it is a real remainder rather than a residue
            # of three different moments.
            "peak_unattributed_gb": (
                peak_bytes - sum(components.values())
            )
            / _GB,
        }
    return aggregate


def peak_phase(phases: dict) -> str | None:
    """The phase whose largest occurrence set the run's high-water mark."""
    if not phases:
        return None
    return max(phases, key=lambda name: phases[name]["max_interval_peak_gb"])


def _maximum(values: list, default=0):
    present = [v for v in values if v is not None]
    return max(present) if present else default


def _total_or_none(runs: list[dict], key: str):
    """Sum a per-sample timing, or None when the timing was not collected."""
    values = [r.get(key) for r in runs]
    return sum(values) if all(v is not None for v in values) else None


def _first_and_steady_draft(runs: list[dict], total_decode: float) -> dict:
    """The drafter's own prefill, and its steady state, kept apart.

    The first draft call of a request projects the whole prompt's context
    feature and fills the draft KV cache with one entry per prompt token; every
    later call adds only the tokens the last verify accepted. The first is O(S)
    and happens once, the second is O(accepted) and happens every step, so a
    single mean over all draft calls describes neither.
    """

    def mean_of(key):
        values = [r.get(key) for r in runs if r.get(key) is not None]
        return statistics.mean(values) if values else None

    stage_totals: dict[str, float] = {}
    for run in runs:
        for name, value in (run.get("first_draft_stage_s") or {}).items():
            if value is not None:
                stage_totals[name] = stage_totals.get(name, 0.0) + value
    samples = len(runs) or 1
    return {
        # First draft call
        "mean_first_draft_forward_s": mean_of("first_draft_forward_s"),
        "total_first_draft_forward_s": _total_or_none(runs, "first_draft_forward_s"),
        "max_first_draft_peak_memory_gb": _maximum(
            [r["first_draft_peak_memory_bytes"] for r in runs]
        )
        / _GB,
        "max_first_draft_transient_gb": _maximum(
            [r["first_draft_transient_bytes"] for r in runs]
        )
        / _GB,
        "max_first_draft_cache_gb": _maximum(
            [r["first_draft_cache_bytes"] for r in runs]
        )
        / _GB,
        "max_first_draft_cache_pre_crop_gb": _maximum(
            [r["first_draft_cache_pre_crop_bytes"] for r in runs]
        )
        / _GB,
        "mean_first_draft_fraction_of_decode": mean_of("first_draft_fraction_of_decode"),
        # The aggregate share, weighted by decode time rather than by request:
        # a long generation amortises the first call, a short one does not.
        "first_draft_share_of_decode": (
            None
            if _total_or_none(runs, "first_draft_forward_s") is None or not total_decode
            else _total_or_none(runs, "first_draft_forward_s") / total_decode
        ),
        "first_draft_stage_s": {
            name: total / samples for name, total in stage_totals.items()
        }
        or None,
        # Steady state
        "mean_steady_draft_calls": statistics.mean(
            r["steady_draft_calls"] for r in runs
        ),
        "mean_steady_draft_forward_s": mean_of("mean_steady_draft_forward_s"),
        "mean_steady_attention_s": mean_of("mean_steady_attention_s"),
        "mean_steady_cache_update_s": mean_of("mean_steady_cache_update_s"),
        "mean_steady_context_kv_projection_s": mean_of(
            "mean_steady_context_kv_projection_s"
        ),
        "mean_steady_context_projection_s": mean_of("mean_steady_context_projection_s"),
        "mean_steady_output_head_s": mean_of("mean_steady_output_head_s"),
        "max_steady_draft_cache_gb": _maximum(
            [r["steady_draft_cache_bytes"] for r in runs]
        )
        / _GB,
        "max_steady_draft_peak_memory_gb": _maximum(
            [r["steady_draft_peak_memory_bytes"] for r in runs]
        )
        / _GB,
        "max_steady_draft_transient_gb": _maximum(
            [r["steady_draft_transient_bytes"] for r in runs]
        )
        / _GB,
        "max_prefill_context_feature_gb": _maximum(
            [r["prefill_context_feature_bytes"] for r in runs]
        )
        / _GB,
        "max_steady_context_feature_gb": _maximum(
            [r["steady_context_feature_bytes"] for r in runs]
        )
        / _GB,
    }


def summarize(
    runs: list[dict],
    block_size: int,
    draft_weight_bytes: int,
    *,
    target_weight_bytes: int = 0,
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
        # Where that peak came from: the operation, decode token and drafter
        # step of the worst interval across every sample. See PeakTracker.
        "peak_site": _peak_site(runs),
        # The pre-fix figure -- per-device maxima summed regardless of whether
        # they coexisted. Equal to peak_memory_gb on one device; larger when the
        # target is sharded, and the gap is pure over-count.
        "peak_memory_sum_device_maxima_gb": _maximum(
            [r.get("peak_memory_sum_device_maxima_bytes") for r in runs]
        )
        / _GB,
        # Allocated is what the tensors occupy; reserved is what the caching
        # allocator holds. nvidia-smi shows reserved plus the CUDA context, so
        # reserved is the figure to reconcile against it.
        "peak_memory_reserved_gb": _maximum(
            [r["peak_memory_reserved_bytes"] for r in runs]
        )
        / _GB,
        "max_target_cache_gb": _maximum([r["target_cache_bytes"] for r in runs]) / _GB,
        # Which operation set each sample's peak, counted over samples.
        "peak_site_histogram": _peak_site_histogram(runs),
        # The target's own weights. The baseline pays these too, so they are
        # the floor both configurations are measured against.
        "target_weight_gb": target_weight_bytes / _GB,
        # Phase-local memory: per phase, a decomposition read at one instant.
        # See aggregate_phases and PhaseMemory.
        "phase_memory": aggregate_phases(runs, target_weight_bytes),
        # The target's prefill activation, measured inside the prefill phase
        # rather than inferred from the run's peak. The baseline pays this too,
        # so it is target-architecture overhead, never drafter overhead.
        "max_target_prefill_transient_gb": _maximum(
            [r.get("target_prefill_transient_bytes") for r in runs]
        )
        / _GB,
    }
    summary["peak_phase"] = peak_phase(summary["phase_memory"])
    if not drafter:
        summary["mean_decode_steps"] = statistics.mean(
            r["num_decode_steps"] for r in runs
        )
        # Not a cost the baseline incurs, but a tensor resident on its device.
        summary["draft_weight_gb"] = draft_weight_bytes / _GB
        summary["memory_budget"] = _budget(summary, drafter=False)
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
    summary.update(_first_and_steady_draft(runs, total_decode))
    # What is DFlash's and what is the target architecture's. Keeping the two
    # apart is the whole point: a target KV cache and a target prefill
    # activation grow with context whether or not a drafter exists.
    summary["overhead_split"] = {
        "dflash_persistent_resident_gb": _maximum(
            [r["dflash_persistent_resident_bytes"] for r in runs]
        )
        / _GB,
        "dflash_draft_weight_gb": draft_weight_bytes / _GB,
        "dflash_draft_kv_gb": max_draft_cache / _GB,
        "dflash_selected_hidden_gb": max_hidden_states / _GB,
        "dflash_context_feature_gb": max_context_feature / _GB,
        "dflash_first_draft_transient_gb": _maximum(
            [r["first_draft_transient_bytes"] for r in runs]
        )
        / _GB,
        "dflash_steady_draft_transient_gb": _maximum(
            [r["steady_draft_transient_bytes"] for r in runs]
        )
        / _GB,
        "target_weight_gb": target_weight_bytes / _GB,
        "target_kv_gb": summary["max_target_cache_gb"],
        "target_prefill_transient_gb": summary["max_target_prefill_transient_gb"],
    }
    # Built last: it reads the drafter terms above, which only exist by now.
    summary["memory_budget"] = _budget(summary, drafter=True)
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
        site = summary.get("peak_site")
        if site:
            row("Peak hit at", describe_peak_site(site))
            if len(site.get("per_device_gb") or []) > 1:
                row(
                    "  peak split over devices",
                    " + ".join(f"{g:.2f}" for g in site["per_device_gb"]) + " GB",
                )
                row(
                    "  sum of per-device maxima",
                    f"{summary['peak_memory_sum_device_maxima_gb']:.2f} GB "
                    f"(over-counts by "
                    f"{summary['peak_memory_sum_device_maxima_gb'] - summary['peak_memory_gb']:.2f})",
                )
            histogram = summary.get("peak_site_histogram") or {}
            if len(histogram) > 1:
                total = sum(histogram.values())
                row(
                    "  peaking operation over samples",
                    ", ".join(
                        f"{name} {count}/{total}"
                        for name, count in list(histogram.items())[:3]
                    ),
                )
        _print_budget(summary)
        _print_phases(summary)

    def _print_phases(summary: dict) -> None:
        """Per phase: what it entered with, what it peaked at, what it kept."""
        phases = summary.get("phase_memory")
        if not phases:
            return
        hot = summary.get("peak_phase")
        print("  -- phase-local memory (GB; in/peak/out are one occurrence)")
        print(f"    {'phase':<34}{'in':>8}{'peak':>8}{'out':>8}{'borrowed':>10}")
        for name, entry in phases.items():
            print(
                f"    {name:<34}"
                f"{entry['peak_allocated_before_gb']:8.2f}"
                f"{entry['max_interval_peak_gb']:8.2f}"
                f"{entry['peak_allocated_after_gb']:8.2f}"
                f"{entry['peak_transient_gb']:10.2f}"
                + ("   <- peak" if name == hot else "")
            )
        if hot:
            components = phases[hot]["peak_components_gb"]
            row(f"  at the peak phase ({hot})", "")
            for key, value in components.items():
                if key == "allocated_bytes":
                    continue
                row(f"    {key.replace('_bytes', '')}", f"{value:7.2f} GB")
            row("    unattributed", f"{phases[hot]['peak_unattributed_gb']:7.2f} GB")

    def _print_budget(summary: dict) -> None:
        """The peak, split into what the run keeps and what it borrows."""
        budget = summary.get("memory_budget")
        if not budget:
            return
        labels = [
            ("target_weight_gb", "target weights"),
            ("target_cache_gb", "target KV cache (max)"),
            ("draft_weight_resident_gb", "draft weights (loaded, unused)"),
            ("draft_weight_gb", "draft weights"),
            ("draft_cache_gb", "draft KV cache (max)"),
            ("target_hidden_states_gb", "target hidden states (max)"),
            ("context_feature_gb", "injected context feature (max)"),
            ("draft_activation_gb", "draft activation (max)"),
        ]
        peak = budget["peak_memory_gb"] or 1.0
        print("  -- memory budget at peak")
        for key, label in labels:
            if key not in budget:
                continue
            value = budget[key]
            row(f"  {label}", f"{value:7.2f} GB   {value / peak * 100:5.1f}%")
        row(
            "  resident subtotal",
            f"{budget['resident_total_gb']:7.2f} GB   "
            f"{budget['resident_total_gb'] / peak * 100:5.1f}%",
        )
        row(
            "  transient (activation, unaccounted)",
            f"{budget['transient_gb']:7.2f} GB   "
            f"{budget['transient_gb'] / peak * 100:5.1f}%",
        )
        row("  = peak allocated", f"{budget['peak_memory_gb']:7.2f} GB")

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
    # The per-term split is printed by the memory budget above; this is the
    # drafter-only subtotal, the figure the context sweep compares across runs.
    row("Total drafter overhead", f"{dflash['draft_overhead_gb']:.2f} GB")
    print("  -- first draft call (the drafter's own prefill)")
    row(
        "Latency (mean per request)",
        f"{(dflash['mean_first_draft_forward_s'] or 0) * 1000:.1f}ms "
        f"({(dflash['mean_first_draft_fraction_of_decode'] or 0) * 100:.1f}% of decode)",
    )
    row(
        "Peak / borrowed",
        f"{dflash['max_first_draft_peak_memory_gb']:.2f} / "
        f"{dflash['max_first_draft_transient_gb']:.2f} GB",
    )
    row(
        "Draft KV left behind (pre-crop)",
        f"{dflash['max_first_draft_cache_gb']:.2f} "
        f"({dflash['max_first_draft_cache_pre_crop_gb']:.2f}) GB",
    )
    stages = dflash.get("first_draft_stage_s") or {}
    if any(v for v in stages.values()):
        row(
            "  stages",
            ", ".join(f"{k} {v * 1000:.1f}ms" for k, v in stages.items()),
        )
    print("  -- steady-state draft calls")
    row("Calls per request (mean)", f"{dflash['mean_steady_draft_calls']:.1f}")
    row(
        "Forward (mean per call)",
        f"{(dflash['mean_steady_draft_forward_s'] or 0) * 1000:.2f}ms",
    )
    for key, label in (
        ("mean_steady_attention_s", "attention"),
        ("mean_steady_cache_update_s", "KV append"),
        ("mean_steady_context_kv_projection_s", "context K/V projection"),
        ("mean_steady_context_projection_s", "context projection"),
        ("mean_steady_output_head_s", "output head"),
    ):
        if dflash.get(key) is not None:
            row(f"  {label}", f"{dflash[key] * 1000:.2f}ms")
    row("Draft KV (max)", f"{dflash['max_steady_draft_cache_gb']:.2f} GB")
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
