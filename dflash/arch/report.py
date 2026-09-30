"""Turning records into the CSVs and tables the analysis actually reads.

Each exporter answers one question and keeps one row shape, so a table can be
joined against another without reshaping. Nothing here recomputes a
measurement; it only selects and flattens what a record already contains, and
where a record has a gap the cell is ``NA`` with the reason carried beside it
rather than a zero.
"""

from __future__ import annotations

import csv
import json
import os

NA = "NA"


def _write_csv(path: str, rows: list[dict], columns: list[str] | None = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return path
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, NA) for k in columns})
    return path


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def phase_rows(record: dict) -> list[dict]:
    """One row per phase per condition: what was live and what it peaked at.

    ``interval_peak_bytes`` is the largest simultaneous total inside the phase.
    The component columns beside it are read at that interval's close, so they
    describe the same instant rather than each component's own maximum.
    """
    rows = []
    for condition in record.get("conditions", []):
        memory = (condition.get("passes") or {}).get("memory") or {}
        median = memory.get("median") or {}
        phases = median.get("phase_memory") or {}
        for name, entry in phases.items():
            for field in ("peak", "first"):
                record_entry = entry.get(field)
                if not record_entry:
                    continue
                components = record_entry.get("components") or {}
                rows.append({
                    "model": condition.get("model_key"),
                    "condition": condition.get("key"),
                    "sweep": condition.get("sweep"),
                    "input_tokens": condition.get("input_tokens"),
                    "block_size": condition.get("block_size"),
                    "max_new_tokens": condition.get("max_new_tokens"),
                    "phase": name,
                    "which": field,
                    "interval_peak_bytes": record_entry.get("interval_peak_bytes"),
                    "allocated_before_bytes": record_entry.get("allocated_before_bytes"),
                    "allocated_after_bytes": record_entry.get("allocated_after_bytes"),
                    "target_attention_kv_bytes":
                        components.get("target_attention_kv_bytes", NA),
                    "target_gdn_recurrent_bytes":
                        components.get("target_gdn_recurrent_bytes", NA),
                    "target_gdn_conv_working_bytes":
                        components.get("target_gdn_conv_working_bytes", NA),
                    "target_gdn_conv_recording_bytes":
                        components.get("target_gdn_conv_recording_bytes", NA),
                    "draft_attention_kv_bytes":
                        components.get("draft_attention_kv_bytes", NA),
                    "draft_weight_bytes": components.get("draft_weight_bytes", NA),
                    "target_selected_hidden_bytes":
                        components.get("target_selected_hidden_bytes", NA),
                    "context_feature_bytes":
                        components.get("context_feature_bytes", NA),
                })
    return rows


def component_rows(record: dict) -> list[dict]:
    """The decode-state split at the end of prefill, one row per condition.

    This is the table research question 1 is read off: how the target's
    attention KV, the GDN state, the drafter's own KV, the selected hidden
    states and the context feature trade off as S grows. Ratios are given
    against two denominators, because "draft KV is 40% of the KV cache" and
    "draft KV is 12% of everything the target carries between steps" are
    different claims and only one of them is usually meant.
    """
    # Two moments, not one. At the end of prefill the drafter has not run, so
    # its KV is zero and a draft-KV ratio taken there is meaningless. In the
    # steady state the prompt-length context feature and the conv recording
    # buffer have been released. Neither moment alone describes the request.
    moments = (
        ("prefill_end", "prefill: rollback/crop"),
        ("first_draft", "decode: first draft forward"),
        ("steady_decode", "decode: draft forward"),
    )
    rows = []
    for condition in record.get("conditions", []):
        memory = (condition.get("passes") or {}).get("memory") or {}
        median = memory.get("median") or {}
        phases = median.get("phase_memory") or {}
        for moment, phase_name in moments:
            entry = (phases.get(phase_name) or {}).get("peak")
            if not entry:
                continue
            components = entry.get("components") or {}
            attention = components.get("target_attention_kv_bytes") or 0
            gdn = (
                (components.get("target_gdn_recurrent_bytes") or 0)
                + (components.get("target_gdn_conv_working_bytes") or 0)
            )
            draft_kv = components.get("draft_attention_kv_bytes") or 0
            target_decode_state = attention + gdn
            rows.append({
                "model": condition.get("model_key"),
                "condition": condition.get("key"),
                "moment": moment,
                "phase": phase_name,
                "input_tokens": condition.get("input_tokens"),
                "block_size": condition.get("block_size"),
                "interval_peak_bytes": entry.get("interval_peak_bytes", NA),
                "target_attention_kv_bytes": attention,
                "target_gdn_state_bytes": gdn,
                # O(S), and held only so a rejected block can be rolled back.
                "target_gdn_conv_recording_bytes":
                    components.get("target_gdn_conv_recording_bytes", NA),
                "target_decode_state_bytes": target_decode_state,
                "draft_kv_bytes": draft_kv,
                "draft_weight_bytes": components.get("draft_weight_bytes", NA),
                "target_selected_hidden_bytes":
                    components.get("target_selected_hidden_bytes", NA),
                "context_feature_bytes":
                    components.get("context_feature_bytes", NA),
                # Denominator 1: the target's attention KV alone.
                "draft_kv_over_target_attention_kv":
                    (draft_kv / attention) if attention else NA,
                # Denominator 2: everything the target carries between steps.
                "draft_kv_over_target_decode_state":
                    (draft_kv / target_decode_state) if target_decode_state else NA,
            })
    return rows


def _first(mapping: dict, *keys, default=NA):
    """The first key present -- new field names, then the pre-v2 ones."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _prompt_columns(condition: dict) -> dict:
    prompt = condition.get("prompt") or {}
    return {
        "prompt_index": condition.get("prompt_index", NA),
        "prompt_task": prompt.get("task", NA),
        "prompt_source_index": prompt.get("source_index", NA),
        "prompt_composed": prompt.get("composed", NA),
        "prompt_token_hash": prompt.get("token_hash", NA),
    }


def _condition_columns(condition: dict) -> dict:
    return {
        "model": condition.get("model_key"),
        "condition": condition.get("key"),
        "sweep": condition.get("sweep"),
        "input_tokens": condition.get("input_tokens"),
        "block_size": condition.get("block_size"),
        "max_new_tokens": condition.get("max_new_tokens"),
        "output_policy": condition.get("output_policy"),
        **_prompt_columns(condition),
    }


def speedup_rows(record: dict) -> list[dict]:
    """DFlash against AR, per condition, with both AR cache policies kept apart.

    Memory columns follow one rule: a DFlash figure and an AR figure sit side
    by side only when they share a definition. ``*_sum_device_maxima`` is the
    per-device maximum over the run, summed, on both paths (DFlash from the
    untracked perf pass). The interval-resolved simultaneous peak exists only
    for DFlash (memory pass) and has no AR counterpart on a sharded target.
    """
    rows = []
    for condition in record.get("conditions", []):
        passes = condition.get("passes") or {}
        dflash = (passes.get("perf") or {}).get("median") or {}
        memory = (passes.get("memory") or {}).get("median") or {}
        base: dict = {
            **_condition_columns(condition),
            "flags": ";".join(condition.get("flags") or []) or "-",
            "dflash_ok": (passes.get("perf") or {}).get("ok", False),
            "dflash_error": (passes.get("perf") or {}).get("error_class", NA),
            "dflash_tpot_s": dflash.get("time_per_output_token_s", NA),
            "dflash_ttft_s": dflash.get("time_to_first_token_s", NA),
            "dflash_decode_s": dflash.get("decode_latency_s", NA),
            "dflash_output_tokens": dflash.get("num_output_tokens", NA),
            "acceptance_rate": dflash.get("acceptance_rate", NA),
            "mean_committed_per_step": dflash.get("mean_committed_per_step", NA),
            "num_verify_steps": dflash.get("num_verify_steps", NA),
            # Must be False: a timing taken after torch.profiler ran in the
            # process is inflated (see trace.PROFILER_USED).
            "dflash_profiler_used_before": (
                ((passes.get("perf") or {}).get("samples") or [{}])[0]
                .get("profiler_used_before", NA)
            ),
            "dflash_verify_mean_s": _first(dflash, "target_verify_mean_s"),
            "dflash_peak_simultaneous_bytes": _first(
                memory, "peak_memory_simultaneous_bytes", "peak_memory_bytes"
            ),
            "dflash_peak_sum_device_maxima_bytes": dflash.get(
                "peak_memory_sum_device_maxima_bytes", NA
            ),
            "dflash_reserved_sum_device_maxima_bytes": dflash.get(
                "peak_reserved_sum_device_maxima_bytes", NA
            ),
        }
        for policy in ("native", "recording"):
            result = condition.get("ar", {}).get(policy) or {}
            median = result.get("median") or {}
            samples = result.get("samples") or [{}]
            base[f"ar_{policy}_ok"] = result.get("ok", False)
            base[f"ar_{policy}_error"] = result.get("error_class", NA)
            base[f"ar_{policy}_tpot_s"] = median.get("time_per_output_token_s", NA)
            base[f"ar_{policy}_ttft_s"] = median.get("time_to_first_token_s", NA)
            base[f"ar_{policy}_peak_sum_device_maxima_bytes"] = median.get(
                "peak_memory_sum_device_maxima_bytes", NA
            )
            base[f"ar_{policy}_reserved_sum_device_maxima_bytes"] = median.get(
                "peak_reserved_sum_device_maxima_bytes", NA
            )
            # Pre-v2 records never set this; their AR ran with the drafter
            # loaded, which "NA" must not be read as contradicting.
            base[f"ar_{policy}_drafter_resident"] = samples[0].get(
                "drafter_resident", NA
            )
            base[f"ar_{policy}_profiler_used_before"] = samples[0].get(
                "profiler_used_before", NA
            )
            speedup = condition.get("speedup", {}).get(policy) or {}
            base[f"speedup_vs_{policy}_tpot"] = speedup.get("tpot_speedup", NA)
            base[f"speedup_vs_{policy}_e2e"] = speedup.get("e2e_speedup", NA)
            base[f"verify_over_ar_{policy}_step"] = speedup.get(
                "verify_over_ar_step", NA
            )
            base[f"memory_overhead_vs_{policy}_bytes"] = speedup.get(
                "peak_memory_sum_device_maxima_bytes_overhead", NA
            )
        rows.append(base)
    return rows


def timing_rows(record: dict) -> list[dict]:
    """Where DFlash's decode time goes, per condition, from the perf pass.

    Totals and per-call means are separate columns; nothing here is a mean
    multiplied back up or a total divided down except the per-step columns,
    which say so. ``unattributed`` is wall time no timer claims.
    """
    rows = []
    for condition in record.get("conditions", []):
        perf = ((condition.get("passes") or {}).get("perf") or {}).get("median") or {}
        if not perf:
            continue
        decode = perf.get("decode_latency_s")
        steps = perf.get("num_verify_steps")
        row = {
            **_condition_columns(condition),
            "decode_latency_s": decode,
            "num_verify_steps": steps,
        }
        for key in (
            "target_verify_total_s", "target_verify_mean_s",
            "draft_forward_total_s", "draft_forward_first_s",
            "draft_forward_steady_mean_s", "draft_logits_total_s",
            "context_feature_decode_total_s", "context_feature_prefill_s",
            "attributed_decode_s", "unattributed_decode_s",
            "unattributed_decode_per_step_s",
        ):
            row[key] = perf.get(key, NA)
        for key in ("target_verify_total_s", "draft_forward_total_s",
                    "draft_logits_total_s", "context_feature_decode_total_s",
                    "unattributed_decode_s"):
            value = perf.get(key)
            row[key[:-2] + "_fraction"] = (
                value / decode if isinstance(value, (int, float)) and decode else NA
            )
        rows.append(row)
    return rows


def device_rows(record: dict) -> list[dict]:
    """Per-GPU memory, under each definition, per condition.

    ``at_aggregate_peak`` is the split of the largest simultaneous total --
    what each device held at that instant, not its own peak.
    ``device_maxima`` is each device's own maximum over the run, which is the
    definition the AR baseline uses; the two are different quantities and are
    never subtracted from one another.
    """
    rows = []
    for condition in record.get("conditions", []):
        passes = condition.get("passes") or {}
        memory = (passes.get("memory") or {}).get("median") or {}
        perf = (passes.get("perf") or {}).get("median") or {}
        native = ((condition.get("ar") or {}).get("native") or {}).get("median") or {}
        at_peak = memory.get("peak_memory_per_device_at_aggregate_peak_bytes")
        if at_peak is None:
            at_peak = memory.get("peak_memory_per_device_bytes") or []
        mem_maxima = memory.get("peak_memory_device_maxima_bytes") or []
        perf_maxima = perf.get("peak_memory_device_maxima_bytes") or []
        perf_reserved = perf.get("peak_reserved_device_maxima_bytes") or []
        ar_maxima = native.get("peak_memory_device_maxima_bytes")
        if ar_maxima is None:
            # Pre-v2 AR rows used this name for per-device maxima.
            ar_maxima = native.get("peak_memory_per_device_bytes") or []
        ar_reserved = native.get("peak_reserved_device_maxima_bytes") or []
        count = max(len(at_peak), len(mem_maxima), len(perf_maxima), len(ar_maxima))

        def cell(values, index):
            return values[index] if index < len(values) else NA

        for index in range(count):
            rows.append({
                "model": condition.get("model_key"),
                "condition": condition.get("key"),
                "device": f"cuda:{index}",
                "dflash_at_aggregate_peak_bytes": cell(at_peak, index),
                "dflash_device_max_memory_pass_bytes": cell(mem_maxima, index),
                "dflash_device_max_perf_pass_bytes": cell(perf_maxima, index),
                "dflash_device_max_reserved_bytes": cell(perf_reserved, index),
                "ar_native_device_max_bytes": cell(ar_maxima, index),
                "ar_native_device_max_reserved_bytes": cell(ar_reserved, index),
                "aggregate_simultaneous_peak_bytes": _first(
                    memory, "peak_memory_simultaneous_bytes", "peak_memory_bytes"
                ),
                "peak_operation": (memory.get("peak_site") or {}).get("operation", NA),
            })
    return rows


def _trace_blocks(condition: dict):
    yield "dflash", ((condition.get("passes") or {}).get("trace") or {})
    yield "ar", (condition.get("ar_trace") or {})


def trace_device_rows(record: dict) -> list[dict]:
    """Per device over the decode window: busy, shard wait, all-idle."""
    rows = []
    for condition in record.get("conditions", []):
        for path, block in _trace_blocks(condition):
            median = block.get("median") or {}
            analysis = median.get("analysis") or {}
            if not analysis.get("ok"):
                continue
            for device, values in analysis["devices"].items():
                rows.append({
                    **_condition_columns(condition),
                    "path": path,
                    "device": device,
                    "decode_window_s": analysis["decode_window_s"],
                    "profiled_decode_latency_s": median.get("profiled_decode_latency_s"),
                    **values,
                })
    return rows


def trace_phase_rows(record: dict) -> list[dict]:
    """Device time per phase and module kind: what a verify is made of."""
    rows = []
    for condition in record.get("conditions", []):
        for path, block in _trace_blocks(condition):
            analysis = (block.get("median") or {}).get("analysis") or {}
            for phase, entry in (analysis.get("phases") or {}).items():
                occurrences = entry.get("occurrences") or 0
                base = {
                    **_condition_columns(condition),
                    "path": path,
                    "phase": phase,
                    "occurrences": occurrences,
                    "phase_device_time_s": entry.get("device_time_s"),
                    "kernels_per_occurrence": entry.get("kernels_per_occurrence"),
                }
                for kind, seconds in (entry.get("module_kind_s") or {}).items():
                    rows.append({
                        **base,
                        "breakdown": "module",
                        "part": kind,
                        "device_time_s": seconds,
                        "per_occurrence_s": seconds / occurrences if occurrences else NA,
                    })
                for category, seconds in (entry.get("kernel_category_s") or {}).items():
                    rows.append({
                        **base,
                        "breakdown": "kernel_category",
                        "part": category,
                        "device_time_s": seconds,
                        "per_occurrence_s": seconds / occurrences if occurrences else NA,
                    })
    return rows


def trace_kernel_rows(record: dict) -> list[dict]:
    """The top kernels of each decode phase, by device time."""
    rows = []
    for condition in record.get("conditions", []):
        for path, block in _trace_blocks(condition):
            analysis = (block.get("median") or {}).get("analysis") or {}
            for phase, entry in (analysis.get("phases") or {}).items():
                for rank, kernel in enumerate(entry.get("top_kernels") or []):
                    rows.append({
                        "model": condition.get("model_key"),
                        "condition": condition.get("key"),
                        "path": path,
                        "phase": phase,
                        "rank": rank,
                        **kernel,
                    })
    return rows


def attention_call_rows(record: dict) -> list[dict]:
    """What each full-attention call was handed, per phase and query width."""
    rows = []
    for condition in record.get("conditions", []):
        for path, block in _trace_blocks(condition):
            for call in (block.get("median") or {}).get("attention_calls") or []:
                mask = call.get("mask")
                rows.append({
                    "model": condition.get("model_key"),
                    "condition": condition.get("key"),
                    "path": path,
                    "phase": call.get("phase"),
                    "query_tokens": call.get("query_tokens"),
                    "mask": json.dumps(mask) if mask is not None else "none",
                    "attn_implementation": call.get("attn_implementation"),
                    "calls": call.get("calls"),
                })
    return rows


def moe_rows(record: dict) -> list[dict]:
    """Per-layer routing for every verify (DFlash) and every AR step."""
    rows = []
    for condition in record.get("conditions", []):
        for path, block in (
            ("dflash", (condition.get("passes") or {}).get("moe") or {}),
            ("ar", condition.get("ar_moe") or {}),
        ):
            if not block.get("ok"):
                continue
            for row in routing_rows({"records": block.get("samples") or []}):
                rows.append({
                    "model": condition.get("model_key"),
                    "condition": condition.get("key"),
                    "path": path,
                    **row,
                })
    return rows


def routing_rows(routing: dict) -> list[dict]:
    """Per-layer MoE routing, one row per layer per captured forward."""
    rows = []
    for entry in routing.get("records", []):
        rows.append({
            "phase": entry.get("phase"),
            "step": entry.get("step", NA),
            "layer": entry.get("layer"),
            "device": entry.get("device"),
            "num_tokens": entry.get("num_tokens"),
            "top_k": entry.get("top_k"),
            "num_experts": entry.get("num_experts"),
            "unique_experts_hit": entry.get("unique_experts_hit"),
            "expert_hit_fraction": entry.get("expert_hit_fraction"),
            "tokens_per_expert_max": entry.get("tokens_per_expert_max"),
            "tokens_per_expert_mean": entry.get("tokens_per_expert_mean"),
            "load_imbalance": entry.get("load_imbalance", NA),
            "router_s": entry.get("router_s", NA),
            "expert_loop_s": entry.get("expert_loop_s", NA),
            "na_reason": entry.get("na_reason", ""),
        })
    return rows


def verify_width_rows(record: dict) -> list[dict]:
    rows = []
    for result in record.get("results", []):
        if not isinstance(result, dict) or "widths" not in result:
            # Not a verify-width record. Several record kinds use "results"
            # as their top-level list, so the exporter identifies its own
            # shape rather than trusting the key.
            return []
        if not result.get("ok"):
            rows.append({
                "model": record.get("model"),
                "input_tokens": result.get("prompt_tokens"),
                "q": NA, "median_s": NA, "na_reason": result.get("na_reason"),
            })
            continue
        for row in result["results"]:
            rows.append({
                "model": record.get("model"),
                "input_tokens": result["prompt_tokens"],
                "cache_policy": result.get("cache_policy"),
                "q": row["q"],
                "median_s": row["median_s"],
                "stdev_s": row["stdev_s"],
                "s_per_token": row["s_per_token"],
                "cost_over_q1": row["cost_over_q1"],
                "marginal_s_per_extra_token": row["marginal_s_per_extra_token"],
                "snapshot_s": result["snapshot_s"],
                "mean_restore_s": result["mean_restore_s"],
                "snapshot_bytes": result["snapshot_bytes"],
            })
    return rows


def rollback_rows(record: dict) -> list[dict]:
    """One row per (accepted count, reference, mixer) of a rollback check."""
    rows = []
    for result in record.get("results", []):
        for reference, entry in result.get("references", {}).items():
            # Tensor kinds, resolved per mixer. The reference-wide lists would
            # attribute "recurrent differs" to the attention rows too, which is
            # the opposite of what this table exists to show.
            per_mixer_kinds: dict = {}
            for row in entry.get("state_rows", []) or []:
                bucket = per_mixer_kinds.setdefault(
                    row.get("mixer"), {"differs": set(), "equal": set()}
                )
                kind = str(row.get("tensor", "")).split("[")[0]
                if row.get("status") == "differs":
                    bucket["differs"].add(kind)
                elif row.get("status") == "equal":
                    bucket["equal"].add(kind)
            for mixer, bucket in entry.get("state_comparison_by_mixer", {}).items():
                kinds = per_mixer_kinds.get(mixer, {"differs": set(), "equal": set()})
                rows.append({
                    "model": record.get("model"),
                    "prompt_tokens": result.get("prompt_tokens"),
                    "block_size": result.get("block_size"),
                    "accepted": result.get("accepted"),
                    "rejected": result.get("rejected"),
                    "reference": reference,
                    "mixer": mixer,
                    "tensors": bucket.get("tensors"),
                    "equal": bucket.get("equal"),
                    "differs": bucket.get("differs"),
                    "max_abs_diff": bucket.get("max_abs_diff"),
                    "differing_tensor_kinds":
                        ";".join(sorted(kinds["differs"])) or "-",
                    "matching_tensor_kinds":
                        ";".join(sorted(kinds["equal"])) or "-",
                    "next_token_argmax_match":
                        entry.get("next_token_argmax_match"),
                    "logits_max_abs_diff": entry.get("logits_max_abs_diff"),
                })
    return rows


EXPORTERS = {
    "phase": phase_rows,
    "component": component_rows,
    "speedup": speedup_rows,
    "timing": timing_rows,
    "device": device_rows,
    "trace_device": trace_device_rows,
    "trace_phase": trace_phase_rows,
    "trace_kernel": trace_kernel_rows,
    "attention_calls": attention_call_rows,
    "moe_routing": moe_rows,
    "verify_width": verify_width_rows,
    "rollback": rollback_rows,
}


def lossless_rows(record: dict) -> list[dict]:
    """How far each condition's run is from lossless: state, logits, output.

    One row per condition and reference. ``incremental`` is the error one
    rejection adds; ``accumulated`` is the stock run's drift from a run that
    only ever folded committed tokens. ``*_floor`` columns come from steps
    where every draft was accepted and are the replay's own numeric noise.
    """
    rows = []
    for condition in record.get("conditions", []):
        block = condition.get("lossless")
        if not block:
            continue
        audit = block.get("audit") or {}

        def agreement(name, field):
            value = block.get(name) or {}
            return value.get(field, NA)

        base = {
            "model": condition.get("model_key"),
            "condition": condition.get("key"),
            "input_tokens": condition.get("input_tokens"),
            "block_size": condition.get("block_size"),
            "max_new_tokens": condition.get("max_new_tokens"),
            "gdn_backend": (record.get("gdn_backend") or {}).get("config_label", NA),
            "audit_follows_perf": block.get("audit_follows_perf", NA),
            "acceptance_stock": block.get("acceptance_rate_stock", NA),
            "acceptance_exact": block.get("acceptance_rate_exact", NA),
            "acceptance_delta": block.get("acceptance_rate_delta", NA),
            "committed_per_step_stock": block.get("mean_committed_per_step_stock", NA),
            "committed_per_step_exact": block.get("mean_committed_per_step_exact", NA),
            "stock_vs_ar_first_divergence": agreement("stock_vs_ar", "first_divergence"),
            "stock_vs_ar_match_rate": agreement("stock_vs_ar", "positional_match_rate"),
            "exact_vs_ar_first_divergence": agreement("exact_vs_ar", "first_divergence"),
            "exact_vs_ar_match_rate": agreement("exact_vs_ar", "positional_match_rate"),
            "stock_vs_exact_first_divergence": agreement("stock_vs_exact", "first_divergence"),
            "steps": audit.get("num_steps", NA),
            "steps_with_rejection": audit.get("steps_with_rejection", NA),
            "rejected_tokens_folded": audit.get("rejected_tokens_folded_total", NA),
        }
        for reference in ("incremental", "accumulated"):
            ref = audit.get(reference)
            if not ref:
                continue
            row = dict(base, reference=reference)

            def stat(subset, metric, field):
                value = ((ref.get(subset) or {}).get(metric) or {})
                return value.get(field, NA)

            row.update({
                "state_rel_err_median": stat("rejecting", "state_rel_err_mean", "median"),
                "state_rel_err_p90": stat("rejecting", "state_rel_err_mean", "p90"),
                "state_rel_err_worst_layer_max": stat("rejecting", "state_rel_err_max", "max"),
                "state_rel_err_floor_median": stat("floor", "state_rel_err_mean", "median"),
                "final_state_rel_err": ref.get("final_state_rel_err_mean", NA),
                "kl_median": stat("rejecting", "kl_exact_to_stock", "median"),
                "kl_max": stat("rejecting", "kl_exact_to_stock", "max"),
                "kl_floor_median": stat("floor", "kl_exact_to_stock", "median"),
                "max_abs_logit_diff_median": stat("rejecting", "max_abs_logit_diff", "median"),
                "argmax_flips": ref.get("argmax_flips", NA),
                "next_token_steps": ref.get("next_token_steps", NA),
                "argmax_flip_rate": ref.get("argmax_flip_rate", NA),
                "first_flip_step": ref.get("first_flip_step", NA),
            })
            rows.append(row)
    return rows


EXPORTERS["lossless"] = lossless_rows


def csv_dir_for(record_path: str) -> str:
    """``<model dir>/csv/<record name>/``: one directory per record.

    A single ``csv/`` per model let every sweep overwrite the last one's
    tables -- a one-condition no-FLA run replaced the full FLA sweep's.
    """
    directory, name = os.path.split(record_path)
    return os.path.join(directory, "csv", os.path.splitext(name)[0])


def export(record: dict, out_dir: str, *, kinds: tuple[str, ...] | None = None) -> dict:
    """Write every applicable CSV for one record. Empty tables are skipped."""
    written = {}
    for name in (kinds or tuple(EXPORTERS)):
        exporter = EXPORTERS.get(name)
        if exporter is None:
            continue
        try:
            rows = exporter(record)
        except (KeyError, TypeError, AttributeError):
            continue
        if rows:
            written[name] = _write_csv(os.path.join(out_dir, f"{name}.csv"), rows)
    return written
