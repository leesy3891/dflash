"""One measured condition: the passes, and what each of them is allowed to say.

A condition is (model, S, block width, output length, output policy). It is
measured in more than one pass because the passes interfere with each other.

``perf``
    No component probe, no per-layer hooks, no allocator reads at phase
    boundaries (``memory_tracking=False``), no synchronizes beyond the ones
    the timing needs. Repeated, median reported with spread. This is the only
    pass whose latency, TPOT and speedup numbers go into a result table, and
    it carries no memory figures: those come from ``memory``.
``memory``
    PeakTracker with per-layer hooks and the component probe installed, so
    every phase boundary reads a resolved split of what was simultaneously
    live. Its latency is recorded but marked, and never mixed into ``perf``;
    the perf/memory TPOT difference is the instrumentation's cost.
``audit`` / ``exact``
    The lossless audit (``lossless.VerifyAudit``). ``audit`` measures, at
    every verify step, how far the stock state is from the state a lossless
    rollback would leave, and then restores the stock state so the run stays
    on ``perf``'s trajectory. ``exact`` keeps the lossless state instead, so
    its acceptance is what the run would have had without the GDN rollback
    defect. Both replay the committed tokens every step: one run each, and
    their timing is never a latency result.
``trace``
    One run under ``torch.profiler`` with phase and module ranges; device
    busy/idle and per-phase, per-module, per-layer device time
    (:mod:`dflash.arch.trace`). Diagnostic only.
``moe``
    One run with :class:`dflash.arch.moe.RoutingCapture` on every MoE layer:
    expert hit counts per verify and per AR step. Diagnostic only.
``counters``
    Hardware counters (DRAM bytes, SM throughput) through Nsight Compute.
    Not implemented: the installed ncu 2024.1.1 fails to profile any kernel
    of this torch build (CUDA 12.9 runtime), even a single-metric sgemm.

Timing fields
-------------
Every timer field says whether it is a total over the run (``*_total_s``) or a
mean per call (``*_mean_s``); the unsuffixed names ``dflash_generate`` returns
mixed the two in one row. ``unattributed_decode_s`` is decode wall time no
timer claims -- noise embedding, crops, the acceptance read-back, accelerate
hooks -- reported, never distributed.

TPOT is ``decode_latency / (output_tokens - 1)`` on both paths: the first
output token comes from prefill in both, so it belongs to TTFT.

The perturbation between ``perf`` and ``memory`` is measured at the
representative condition rather than assumed small, and acceptance is compared
across passes: if the probe changed which tokens were accepted, something is
wrong with the probe, not with the model.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, asdict

import torch

from ..model import dflash_generate, module_bytes
from . import ar, lossless, probe as probe_module, statemem, taxonomy

PASS_PERF = "perf"
PASS_MEMORY = "memory"
PASS_AUDIT = "audit"
PASS_EXACT = "exact"
PASS_TRACE = "trace"
PASS_MOE = "moe"
PASS_COUNTERS = "counters"

# Passes cmd_sweep can run per condition.
DFLASH_PASSES = (PASS_PERF, PASS_MEMORY, PASS_AUDIT, PASS_EXACT, PASS_TRACE, PASS_MOE)

OUTPUT_SHAPE_CONTROLLED = "shape_controlled"
OUTPUT_NATURAL = "natural"


@dataclass
class Condition:
    """One point in the sweep. Axes are swept separately, never crossed."""

    model_key: str
    input_tokens: int
    block_size: int
    max_new_tokens: int
    output_policy: str = OUTPUT_SHAPE_CONTROLLED
    sweep: str = "sequence"
    repeats: int = 3
    warmup: int = 1
    note: str | None = None
    # Which of the K prompts at this S. None on records from before prompt
    # sets existed, and then left out of the key so old keys still match.
    prompt_index: int | None = None

    @property
    def ignore_eos(self) -> bool:
        return self.output_policy == OUTPUT_SHAPE_CONTROLLED

    def key(self) -> str:
        key = (
            f"{self.model_key}/S{self.input_tokens}/b{self.block_size}"
            f"/o{self.max_new_tokens}/{self.output_policy}"
        )
        return key if self.prompt_index is None else f"{key}/p{self.prompt_index}"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PassResult:
    pass_name: str
    ok: bool
    samples: list = field(default_factory=list)
    median: dict = field(default_factory=dict)
    error: str | None = None
    error_class: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _acceptance(stats) -> dict:
    """Acceptance, kept in the two forms that answer different questions."""
    proposed = getattr(stats, "num_proposed_tokens", 0) or 0
    accepted = getattr(stats, "num_accepted_tokens", 0) or 0
    steps = getattr(stats, "num_verify_steps", 0) or 0
    committed = getattr(stats, "acceptance_lengths", []) or []
    return {
        # accepted draft tokens / proposed draft tokens
        "acceptance_rate": (accepted / proposed) if proposed else None,
        "num_accepted_tokens": accepted,
        "num_proposed_tokens": proposed,
        # committed = accepted + the correction or bonus token, clipped to
        # what the output actually took (EOS or the length cap).
        "mean_committed_per_step": (
            sum(committed) / len(committed) if committed else None
        ),
        "num_verify_steps": steps,
        "committed_per_step": committed,
    }


def _visible_devices() -> range:
    return range(torch.cuda.device_count()) if torch.cuda.is_available() else range(0)


def _fresh_allocator() -> None:
    """Start a measurement from the same allocator state every time.

    Without this the reserved pool and its fragmentation carry over from
    whatever ran before -- the other AR policy, the previous condition -- and
    an OOM becomes a property of run order rather than of the condition.
    """
    import gc

    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    for index in _visible_devices():
        try:
            torch.cuda.reset_peak_memory_stats(index)
        except RuntimeError:
            pass


def _profiler_used() -> bool:
    """Whether this process has run torch.profiler; see trace.PROFILER_USED."""
    from . import trace

    return trace.PROFILER_USED


def _reset_peaks() -> None:
    for index in _visible_devices():
        try:
            torch.cuda.reset_peak_memory_stats(index)
        except RuntimeError:
            pass


def _device_peaks() -> dict:
    """Per-device maxima since the last reset: allocated and reserved."""
    allocated, reserved = [], []
    for index in _visible_devices():
        try:
            allocated.append(torch.cuda.max_memory_allocated(index))
            reserved.append(torch.cuda.max_memory_reserved(index))
        except RuntimeError:
            allocated.append(0)
            reserved.append(0)
    return {
        "peak_memory_device_maxima_bytes": allocated,
        "peak_memory_sum_device_maxima_bytes": sum(allocated),
        "peak_reserved_device_maxima_bytes": reserved,
        "peak_reserved_sum_device_maxima_bytes": sum(reserved),
    }


def _allocator_snapshot() -> list[dict]:
    """Where the allocator stood, per device. Taken when a run OOMs.

    ``reserved - allocated`` is memory the caching allocator holds but no
    tensor uses; ``inactive_split_bytes`` is the part of that locked in
    partially used blocks, i.e. fragmentation. Together they separate "the
    condition does not fit" from "the allocator could not place it".
    """
    out = []
    for index in _visible_devices():
        try:
            stats = torch.cuda.memory_stats(index)
            free, total = torch.cuda.mem_get_info(index)
        except RuntimeError as exc:
            out.append({"device": f"cuda:{index}", "error": repr(exc)})
            continue
        out.append({
            "device": f"cuda:{index}",
            "allocated_bytes": stats.get("allocated_bytes.all.current"),
            "reserved_bytes": stats.get("reserved_bytes.all.current"),
            "peak_allocated_bytes": stats.get("allocated_bytes.all.peak"),
            "peak_reserved_bytes": stats.get("reserved_bytes.all.peak"),
            "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
            "num_alloc_retries": stats.get("num_alloc_retries"),
            "num_ooms": stats.get("num_ooms"),
            "device_free_bytes": free,
            "device_total_bytes": total,
        })
    return out


def _oom_result(pass_name: str, exc: BaseException) -> PassResult:
    snapshot = _allocator_snapshot()
    torch.cuda.empty_cache()
    result = PassResult(
        pass_name=pass_name, ok=False,
        error=str(exc)[:600], error_class="OutOfMemoryError",
    )
    result.median = {"allocator_at_oom": snapshot}
    return result


def _timing_fields(stats) -> dict:
    """Timer fields, each named for whether it is a total or a mean."""
    steps = getattr(stats, "num_verify_steps", 0) or 0
    target = stats.target_forward_s
    first_draft = stats.first_draft_forward_s
    draft_total = getattr(stats, "draft_forward_s", None)
    logits = getattr(stats, "draft_logits_s", None)
    decode_feature = getattr(stats, "decode_context_feature_s", None)
    out = {
        "target_verify_total_s": target,
        "target_verify_mean_s": (target / steps) if target is not None and steps else None,
        "draft_forward_total_s": draft_total,
        "draft_forward_first_s": first_draft,
        "draft_forward_steady_mean_s": stats.mean_steady_draft_forward_s,
        "draft_logits_total_s": logits,
        "context_feature_prefill_s": getattr(stats, "prefill_context_feature_s", None),
        "context_feature_decode_total_s": decode_feature,
    }
    parts = (target, draft_total, logits, decode_feature)
    if all(part is not None for part in parts):
        attributed = sum(parts)
        unattributed = stats.decode_latency - attributed
        out["attributed_decode_s"] = attributed
        out["unattributed_decode_s"] = unattributed
        out["unattributed_decode_per_step_s"] = (
            unattributed / steps if steps else None
        )
    return out


def _stats_to_row(stats, *, pass_name: str) -> dict:
    n = stats.num_output_tokens
    tracked = getattr(stats, "memory_tracking", True)
    row = {
        "pass": pass_name,
        "num_input_tokens": stats.num_input_tokens,
        "num_output_tokens": n,
        "time_to_first_token_s": stats.time_to_first_token,
        "decode_latency_s": stats.decode_latency,
        "total_latency_s": stats.total_latency,
        # Same denominator as the AR baseline: the first output token is
        # prefill's in both paths. dflash_generate's own figure divides by n.
        "time_per_output_token_s": (
            stats.decode_latency / (n - 1) if n > 1 else None
        ),
        "time_per_output_token_over_n_s": stats.time_per_output_token,
        "memory_tracking": tracked,
        "layer_hooks": getattr(stats, "layer_hooks", None),
        "profiler_used_before": _profiler_used(),
        "first_draft_cache_bytes": stats.first_draft_cache_bytes,
        "steady_draft_cache_bytes": stats.steady_draft_cache_bytes,
        "draft_weight_bytes": stats.draft_weight_bytes,
        **_timing_fields(stats),
    }
    if tracked:
        row.update({
            # The largest simultaneous total, interval-resolved.
            "peak_memory_simultaneous_bytes": stats.peak_memory_bytes,
            # The per-device split *at* that instant. Not a per-device peak.
            "peak_memory_per_device_at_aggregate_peak_bytes":
                stats.peak_memory_per_device_bytes,
            # Per-device maxima over the run: the AR baseline's definition.
            "peak_memory_device_maxima_bytes":
                getattr(stats, "peak_memory_device_maxima_bytes", None),
            "peak_memory_sum_device_maxima_bytes":
                stats.peak_memory_sum_device_maxima_bytes,
            "peak_site": stats.peak_site,
        })
    row.update(_run_peaks(tracked))
    row.update(_acceptance(stats))
    # Kept so outputs can be compared token by token across passes and
    # against AR; greedy decoding makes that comparison meaningful.
    row["output_token_ids"] = (
        stats.output_ids[0, stats.num_input_tokens:].tolist()
    )
    if pass_name == PASS_MEMORY:
        row["phase_memory"] = stats.phase_memory
    return row


def _run_peaks(tracked: bool) -> dict:
    """Allocator maxima over the run, read after it -- when they mean that.

    Read after the run, outside every timed region, so they cost the timing
    nothing. But PeakTracker resets the allocator's peak counters at every
    boundary, so after a tracked run they cover only the last interval; that
    is also why ``dflash_generate``'s own ``peak_memory_reserved_bytes`` is
    not carried into rows. With tracking on, allocated maxima come from the
    tracker (above) and the reserved peak is not available.
    """
    if tracked:
        return {
            "peak_reserved_device_maxima_bytes": None,
            "peak_reserved_na_reason": (
                "PeakTracker resets the allocator peak counters at every "
                "boundary; reserved peaks are read on the untracked perf pass"
            ),
        }
    return _device_peaks()


def _median_row(rows: list[dict]) -> dict:
    """Median over repeats for numeric fields, with spread kept beside it."""
    if not rows:
        return {}
    numeric = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    out: dict = {key: rows[0][key] for key in rows[0] if key not in numeric}
    for key in numeric:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        out[key] = statistics.median(values)
        out[f"{key}__min"] = min(values)
        out[f"{key}__max"] = max(values)
        out[f"{key}__stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    out["num_repeats"] = len(rows)
    return out


def run_dflash(
    target,
    draft,
    input_ids: torch.Tensor,
    condition: Condition,
    *,
    stop_token_ids: list[int] | None,
    pass_name: str,
    target_mixers: list[str] | None = None,
    draft_mixers: list[str] | None = None,
    hidden_states: str = "selective",
) -> PassResult:
    """Run one condition through DFlash, ``repeats`` times, and reduce."""
    component_probe = (
        probe_module.ComponentProbe(
            target_mixers=target_mixers, draft_mixers=draft_mixers
        )
        if pass_name == PASS_MEMORY
        else None
    )
    stop = None if condition.ignore_eos else stop_token_ids
    audit_mode = {
        PASS_AUDIT: lossless.MEASURE, PASS_EXACT: lossless.CORRECT
    }.get(pass_name)
    # Greedy and deterministic, and each run replays every block: once.
    runs = 1 if audit_mode else condition.warmup + condition.repeats
    skip = 0 if audit_mode else condition.warmup

    # Only the memory pass pays for allocator reads at phase boundaries and
    # per-layer hooks; the AR baseline pays neither, so a timing pass with
    # them on would charge DFlash for its own instrumentation.
    tracked = pass_name == PASS_MEMORY
    rows: list[dict] = []
    # Once per pass, not per repeat: emptying the cache before a timed repeat
    # would put cudaMalloc calls back into the decode loop. The warmup run
    # re-warms the pool, so every pass starts from the same state whatever
    # ran before it.
    _fresh_allocator()
    try:
        for index in range(runs):
            audit = lossless.VerifyAudit(audit_mode) if audit_mode else None
            _reset_peaks()
            torch.cuda.synchronize()
            stats = dflash_generate(
                draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=condition.max_new_tokens,
                stop_token_ids=stop,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                block_size=condition.block_size,
                return_stats=True,
                profile_draft_memory=pass_name == PASS_MEMORY,
                profile_draft_stages=pass_name != PASS_PERF,
                hidden_states=hidden_states,
                component_probe=component_probe,
                verify_audit=audit,
                memory_tracking=tracked,
                watch_layers=tracked,
            )
            if index >= skip:
                row = _stats_to_row(stats, pass_name=pass_name)
                if audit is not None:
                    row["lossless_audit"] = audit.summary()
                    row["lossless_audit_steps"] = audit.steps
                rows.append(row)
    except torch.cuda.OutOfMemoryError as exc:
        return _oom_result(pass_name, exc)
    except Exception as exc:  # noqa: BLE001 - a failed condition is a result
        return PassResult(
            pass_name=pass_name, ok=False,
            error=f"{type(exc).__name__}: {exc}"[:600],
            error_class=type(exc).__name__,
        )
    return PassResult(
        pass_name=pass_name, ok=True, samples=rows, median=_median_row(rows)
    )


def run_ar(
    target,
    input_ids: torch.Tensor,
    condition: Condition,
    *,
    stop_token_ids: list[int] | None,
    cache_policy: str,
    target_mixers: list[str] | None = None,
    drafter_resident: bool = False,
) -> PassResult:
    """The AR baseline at the same condition.

    ``cmd_sweep`` runs this before the drafter is loaded, so the peak here
    contains the target's weights, its cache and its activations and nothing
    else, and subtracting it from a DFlash peak leaves the drafter's
    overhead. ``drafter_resident`` is recorded rather than assumed: a caller
    that runs it with the drafter on the card gets a row that says so.
    """
    rows: list[dict] = []
    _fresh_allocator()
    try:
        for index in range(condition.warmup + condition.repeats):
            _reset_peaks()
            result = ar.generate(
                target,
                input_ids,
                max_new_tokens=condition.max_new_tokens,
                stop_token_ids=stop_token_ids,
                cache_policy=cache_policy,
                ignore_eos=condition.ignore_eos,
                mixers=target_mixers,
            )
            peaks = _device_peaks()
            if index >= condition.warmup:
                rows.append({
                    "pass": PASS_PERF,
                    "cache_policy": cache_policy,
                    "num_input_tokens": result.num_input_tokens,
                    "num_output_tokens": result.num_output_tokens,
                    "time_to_first_token_s": result.time_to_first_token_s,
                    "decode_latency_s": result.decode_latency_s,
                    "total_latency_s": result.total_latency_s,
                    "time_per_output_token_s": result.time_per_output_token_s,
                    "stopped_naturally": result.stopped_naturally,
                    "output_token_ids": list(result.output_ids),
                    "hit_length_cap": result.hit_length_cap,
                    "drafter_resident": drafter_resident,
                    "profiler_used_before": _profiler_used(),
                    # Per-device maxima over the whole AR run, allocated and
                    # reserved -- the same definition as the DFlash perf
                    # pass's fields of the same name. Not interval-resolved;
                    # on a single device the sum and the simultaneous peak
                    # coincide.
                    **peaks,
                    "target_weight_bytes_per_device":
                        ar.weight_bytes_per_device(target),
                })
    except torch.cuda.OutOfMemoryError as exc:
        return _oom_result(PASS_PERF, exc)
    except Exception as exc:  # noqa: BLE001
        return PassResult(
            pass_name=PASS_PERF, ok=False,
            error=f"{type(exc).__name__}: {exc}"[:600],
            error_class=type(exc).__name__,
        )
    return PassResult(
        pass_name=PASS_PERF, ok=True, samples=rows, median=_median_row(rows)
    )


def compare_passes(perf: PassResult, memory: PassResult) -> dict:
    """How much the probe moved the numbers, and whether it changed the run.

    Acceptance is the check that matters. Latency is expected to shift -- the
    probe walks the caches at every phase boundary -- but if the accepted
    token counts differ between passes then the instrumentation is not
    passive and nothing downstream of it can be trusted.
    """
    if not (perf.ok and memory.ok and perf.median and memory.median):
        return {"comparable": False}
    def get(result, key):
        return result.median.get(key)

    perf_tpot = get(perf, "time_per_output_token_s")
    mem_tpot = get(memory, "time_per_output_token_s")
    return {
        "comparable": True,
        "perf_tpot_s": perf_tpot,
        "memory_tpot_s": mem_tpot,
        "tpot_perturbation": (
            (mem_tpot - perf_tpot) / perf_tpot
            if perf_tpot and mem_tpot else None
        ),
        "perf_accepted": get(perf, "num_accepted_tokens"),
        "memory_accepted": get(memory, "num_accepted_tokens"),
        "perf_output_tokens": get(perf, "num_output_tokens"),
        "memory_output_tokens": get(memory, "num_output_tokens"),
        "acceptance_identical": (
            get(perf, "num_accepted_tokens") == get(memory, "num_accepted_tokens")
            and get(perf, "num_output_tokens") == get(memory, "num_output_tokens")
        ),
    }


def speedup(dflash: PassResult, baseline: PassResult) -> dict:
    """DFlash against AR, on the metrics that are legitimately comparable."""
    if not (dflash.ok and baseline.ok):
        return {
            "available": False,
            "reason": (
                f"dflash={'ok' if dflash.ok else dflash.error_class}, "
                f"ar={'ok' if baseline.ok else baseline.error_class}"
            ),
        }
    d, b = dflash.median, baseline.median
    out = {"available": True}
    for label, key in (
        ("tpot", "time_per_output_token_s"),
        ("decode", "decode_latency_s"),
        ("e2e", "total_latency_s"),
        ("ttft", "time_to_first_token_s"),
    ):
        dv, bv = d.get(key), b.get(key)
        out[f"{label}_dflash_s"] = dv
        out[f"{label}_ar_s"] = bv
        # Speedup is AR over DFlash for per-token and latency metrics alike;
        # TTFT is reported as a ratio too but is not a speedup, since DFlash's
        # prefill is the target's own plus the context-feature build.
        out[f"{label}_speedup"] = (bv / dv) if dv and bv else None
    out["output_tokens_dflash"] = d.get("num_output_tokens")
    out["output_tokens_ar"] = b.get("num_output_tokens")
    out["output_tokens_match"] = (
        d.get("num_output_tokens") == b.get("num_output_tokens")
    )
    # Per-step cost, which acceptance does not move: one verify of a block
    # against one AR step. Along S this is the content-independent view --
    # TPOT divides by committed tokens, which the prompt decides.
    verify = d.get("target_verify_mean_s")
    ar_step = b.get("time_per_output_token_s")
    out["verify_mean_s"] = verify
    out["verify_over_ar_step"] = (verify / ar_step) if verify and ar_step else None
    # Memory, on the one definition both paths share: per-device maxima over
    # the run, from untracked runs, with the drafter absent from AR.
    for key in ("peak_memory_sum_device_maxima_bytes",
                "peak_reserved_sum_device_maxima_bytes"):
        dv, bv = d.get(key), b.get(key)
        out[f"{key}_dflash"] = dv
        out[f"{key}_ar"] = bv
        out[f"{key}_overhead"] = (dv - bv) if dv is not None and bv is not None else None
    samples = baseline.samples or [{}]
    out["ar_drafter_resident"] = samples[0].get("drafter_resident")
    return out


def run_trace(
    target,
    draft,
    input_ids: torch.Tensor,
    condition: Condition,
    *,
    stop_token_ids: list[int] | None,
    trace_path: str,
    hidden_states: str = "selective",
    warmup: bool = True,
    max_steps: int = 32,
) -> PassResult:
    """One DFlash run under the profiler; see :mod:`dflash.arch.trace`."""
    stop = None if condition.ignore_eos else stop_token_ids

    def call(phase_callback):
        return dflash_generate(
            draft, target=target, input_ids=input_ids,
            max_new_tokens=condition.max_new_tokens, stop_token_ids=stop,
            temperature=0.0, top_p=1.0, top_k=0,
            block_size=condition.block_size, return_stats=True,
            profile_draft_stages=False, hidden_states=hidden_states,
            memory_tracking=False, watch_layers=False,
            phase_callback=phase_callback,
        )

    return _traced(PASS_TRACE, call, target, trace_path, warmup,
                   "decode: target verify", max_steps)


def run_ar_trace(
    target,
    input_ids: torch.Tensor,
    condition: Condition,
    *,
    stop_token_ids: list[int] | None,
    trace_path: str,
    cache_policy: str = ar.NATIVE,
    warmup: bool = True,
    max_steps: int = 32,
) -> PassResult:
    """The AR baseline under the same profiler, for the same busy/idle split."""
    def call(phase_callback):
        return ar.generate(
            target, input_ids, max_new_tokens=condition.max_new_tokens,
            stop_token_ids=stop_token_ids, cache_policy=cache_policy,
            ignore_eos=condition.ignore_eos, phase_callback=phase_callback,
        )

    return _traced(PASS_TRACE, call, target, trace_path, warmup,
                   "decode: ar step", max_steps)


def _traced(pass_name, call, target, trace_path, warmup, step_phase,
            max_steps) -> PassResult:
    from . import trace

    _fresh_allocator()
    try:
        if warmup:
            call(None)
        profiled = trace.profile_call(
            call, target=target, trace_path=trace_path,
            step_phase=step_phase, max_steps=max_steps,
        )
    except torch.cuda.OutOfMemoryError as exc:
        return _oom_result(pass_name, exc)
    except Exception as exc:  # noqa: BLE001
        return PassResult(
            pass_name=pass_name, ok=False,
            error=f"{type(exc).__name__}: {exc}"[:600],
            error_class=type(exc).__name__,
        )
    result = profiled.pop("result")
    decode = getattr(result, "decode_latency", None)
    if decode is None:
        decode = getattr(result, "decode_latency_s", None)
    steps = getattr(result, "num_verify_steps", None)
    profiled["profiled_decode_latency_s"] = decode
    profiled["num_verify_steps"] = steps
    profiled["num_output_tokens"] = getattr(result, "num_output_tokens", None)
    return PassResult(pass_name=pass_name, ok=True, samples=[], median=profiled)


def run_moe(
    target,
    draft,
    input_ids: torch.Tensor,
    condition: Condition,
    *,
    stop_token_ids: list[int] | None,
    hidden_states: str = "selective",
    keep_expert_ids: bool = True,
) -> PassResult:
    """Expert routing per verify step, from one DFlash run.

    Shape only: ``time_parts=False``, because the capture's own sync-bracketed
    timings are not comparable with anything else in the record. The time the
    router and expert loop take comes from the ``trace`` pass, whose module
    ranges split an MoE FFN into router, routed experts and shared expert
    without a synchronize.
    """
    stop = None if condition.ignore_eos else stop_token_ids
    return _routed(
        lambda cb: dflash_generate(
            draft, target=target, input_ids=input_ids,
            max_new_tokens=condition.max_new_tokens, stop_token_ids=stop,
            temperature=0.0, top_p=1.0, top_k=0,
            block_size=condition.block_size, return_stats=True,
            profile_draft_stages=False, hidden_states=hidden_states,
            memory_tracking=False, watch_layers=False, phase_callback=cb,
        ),
        target, "decode: target verify", keep_expert_ids,
    )


def run_ar_moe(
    target,
    input_ids: torch.Tensor,
    condition: Condition,
    *,
    stop_token_ids: list[int] | None,
    keep_expert_ids: bool = True,
) -> PassResult:
    """The AR baseline's routing: what one token per step hits."""
    return _routed(
        lambda cb: ar.generate(
            target, input_ids, max_new_tokens=condition.max_new_tokens,
            stop_token_ids=stop_token_ids, cache_policy=ar.NATIVE,
            ignore_eos=condition.ignore_eos, phase_callback=cb,
        ),
        target, "decode: ar step", keep_expert_ids,
    )


def _routed(call, target, step_phase: str, keep_expert_ids: bool) -> PassResult:
    from . import moe

    capture = moe.RoutingCapture(target, time_parts=False)
    if not capture._moe_layers:
        return PassResult(
            pass_name=PASS_MOE, ok=False,
            error="target has no MoE layers", error_class="NotApplicable",
        )
    steps = {"n": -1}

    def callback(label, site):
        if label == step_phase:
            steps["n"] += 1
        capture.phase = label or "unset"
        capture.step = steps["n"] if (label or "").startswith("decode:") else None

    _fresh_allocator()
    try:
        with capture:
            call(callback)
    except torch.cuda.OutOfMemoryError as exc:
        return _oom_result(PASS_MOE, exc)
    except Exception as exc:  # noqa: BLE001
        return PassResult(
            pass_name=PASS_MOE, ok=False,
            error=f"{type(exc).__name__}: {exc}"[:600],
            error_class=type(exc).__name__,
        )
    records = []
    for record in capture.records:
        record = dict(record)
        record.pop("routing_weights", None)
        if not keep_expert_ids:
            record.pop("top_k_expert_ids", None)
        records.append(record)
    return PassResult(
        pass_name=PASS_MOE, ok=True, samples=records,
        median={"summary": capture.summary(), "step_phase": step_phase},
    )


def _tokens(result: PassResult | dict | None) -> list[int] | None:
    if result is None:
        return None
    if isinstance(result, dict):
        result = PassResult(**result)
    if not (result.ok and result.samples):
        return None
    return result.samples[0].get("output_token_ids")


def lossless_comparison(passes: dict, ar_results: dict) -> dict:
    """The output-level view of the lossless audit, for one condition.

    For greedy decoding a lossless speculative decoder emits exactly what the
    target alone emits, so AR (native cache) is the reference output.
    ``exact`` departing from AR measures kernel-width numerics only; ``perf``
    departing further is the rollback defect's share. ``audit`` must match
    ``perf`` token for token or the audit perturbed the run.
    """
    ar_tokens = _tokens(ar_results.get(ar.NATIVE))
    perf_tokens = _tokens(passes.get(PASS_PERF))
    audit_tokens = _tokens(passes.get(PASS_AUDIT))
    exact_tokens = _tokens(passes.get(PASS_EXACT))
    out: dict = {}
    if ar_tokens and perf_tokens:
        out["stock_vs_ar"] = lossless.token_agreement(ar_tokens, perf_tokens)
    if ar_tokens and exact_tokens:
        out["exact_vs_ar"] = lossless.token_agreement(ar_tokens, exact_tokens)
    if perf_tokens and exact_tokens:
        out["stock_vs_exact"] = lossless.token_agreement(exact_tokens, perf_tokens)
    if perf_tokens and audit_tokens:
        agreement = lossless.token_agreement(perf_tokens, audit_tokens)
        out["audit_follows_perf"] = agreement["identical"]

    def median(name, key):
        result = passes.get(name)
        if not result or not result.get("ok"):
            return None
        return result["median"].get(key)

    for key in ("acceptance_rate", "mean_committed_per_step"):
        stock, exact = median(PASS_PERF, key), median(PASS_EXACT, key)
        out[f"{key}_stock"] = stock
        out[f"{key}_exact"] = exact
        out[f"{key}_delta"] = (
            stock - exact if stock is not None and exact is not None else None
        )
    audit_pass = passes.get(PASS_AUDIT)
    if audit_pass and audit_pass.get("ok"):
        out["audit"] = audit_pass["samples"][0].get("lossless_audit")
    exact_pass = passes.get(PASS_EXACT)
    if exact_pass and exact_pass.get("ok"):
        out["exact_incremental"] = exact_pass["samples"][0].get("lossless_audit")
    return out
