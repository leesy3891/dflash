"""The GPU timeline pass: which device ran what, when, and who was idle.

Nothing in the ``perf`` or ``memory`` passes looks at the device side of a run.
They time regions with CUDA events on ``cuda:0`` and read allocator counters;
neither says whether a device was busy, and neither can split a verify into
the mixer, FFN and output-head work that make it up. This pass runs a
condition once under ``torch.profiler`` (CUPTI activity tracing, no hardware
counters) and derives, per device, over the decode window:

``busy_s``
    Union of every kernel, memcpy and memset interval on that device. A
    union, not a sum: kernels on one device can overlap across streams, and
    a sum can exceed the wall time it is meant to explain.
``idle_shard_wait_s``
    Time this device was idle while another device had work. On a
    layer-sharded target at B=1 this is structural: while device *i* runs
    its layers the others have nothing to do.
``idle_all_s``
    Time *no* device had work: host launch overhead, a host-device sync
    (the per-step ``.item()`` of the acceptance count), Python between
    kernels. Identical for every device by construction.

and, per phase, per module kind and per layer, the device time kernels
launched from inside that range took. Attribution is by launch: a kernel
belongs to the phase and module whose host range enclosed its launch call,
joined through CUPTI's correlation id. That is exact for what launched it and
says nothing about queueing, which is what the idle split is for.

What this pass is not. It is one profiled run, and the profiler adds host
time to every launch, so host-bound gaps (``idle_all_s``) are *overstated*
relative to the unprofiled ``perf`` pass; the record carries the profiled
decode time beside the ``perf`` one so the size of that distortion is visible.
Kernel durations themselves are not inflated by activity tracing. Busy time
is not SM occupancy and not bandwidth: a kernel that keeps one SM busy counts
the same as one that saturates DRAM. That question needs hardware counters
(Nsight Compute), which the installed ncu cannot collect on this torch build.
"""

from __future__ import annotations

import bisect
import gzip
import json
import os
import re
from collections import Counter, defaultdict

import torch

from . import events as events_module
from . import taxonomy

PHASE_PREFIX = "phase/"

# Set once torch.profiler has run in this process. It leaves every later
# kernel launch slower for the life of the process (AR TPOT +13% on Qwen3-8B,
# +24% on Qwen3.5-35B-A3B), so timing rows record it and cmd_sweep runs every
# timing pass first.
PROFILER_USED = False
MODULE_PREFIX = "mod/"

# Kernel name -> coarse category. Only used for the top-kernel tables; the
# primary breakdown is by module range, which does not depend on names.
_KERNEL_CATEGORIES = (
    ("attention_flash", re.compile(r"flash|fmha_.*flash", re.I)),
    ("attention_mem_efficient", re.compile(r"efficient_attention|fmha|mem_eff|cutlassF", re.I)),
    ("attention_math_softmax", re.compile(r"softmax", re.I)),
    ("gdn_or_conv", re.compile(
        r"chunk_|fused_recurrent|gated_delta|delta_rule|l2norm|causal_conv|conv1d|"
        r"recompute_w_u|solve_tril|fwd_kernel_o|chunk_scaled_dot", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|cublas|splitk|matmul|_mma_|sm80_|sm86_", re.I)),
    ("index_scatter_gather", re.compile(r"index|scatter|gather|nonzero|bincount|topk|sort", re.I)),
    ("reduce", re.compile(r"reduce", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|copy|cat|fill", re.I)),
)


def kernel_category(name: str) -> str:
    for label, pattern in _KERNEL_CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Host ranges
# ---------------------------------------------------------------------------

class PhaseRanges:
    """A ``phase_callback`` that keeps one ``record_function`` open per phase.

    ``dflash_generate`` and ``ar.generate`` call it at every phase boundary
    with the new label, and once with ``None`` at the end.
    """

    def __init__(self) -> None:
        self._open = None
        self.current: str | None = None
        self.counts: Counter = Counter()
        # Ranges are opened (and counted) only while the profiler records.
        self.recording = False

    def __call__(self, label, site) -> None:
        if self._open is not None:
            self._open.__exit__(None, None, None)
            self._open = None
        self.current = label
        if label is None or not self.recording:
            return
        self.counts[label] += 1
        scope = torch.autograd.profiler.record_function(PHASE_PREFIX + label)
        scope.__enter__()
        self._open = scope


class ModuleRanges:
    """``record_function`` ranges around every target mixer, FFN and head.

    Labels are ``mod/<role>/L<index>/<kind>`` so a kernel can be attributed to
    a layer and to what that layer is, and ``mod/target/lm_head`` for the
    output head. MoE FFNs additionally get their router, routed experts and
    shared expert as nested ranges.
    """

    def __init__(self, target, *, attention_probe=None, active=lambda: True) -> None:
        self._active = active
        self._handles: list = []
        self._stacks: dict[int, list] = defaultdict(list)
        spec = taxonomy.describe(target, role="target")
        layers = taxonomy.decoder_layers(target)
        for entry in spec["layers"]:
            layer = layers[entry["index"]]
            index = entry["index"]
            if entry.get("mixer_module"):
                mixer = layer._modules[entry["mixer_module"]]
                self._wrap(mixer, f"target/L{index}/{entry['mixer']}")
                if attention_probe is not None and entry["mixer"] in (
                    taxonomy.MIXER_FULL, taxonomy.MIXER_SLIDING
                ):
                    self._handles.append(mixer.register_forward_pre_hook(
                        attention_probe.hook(index), with_kwargs=True
                    ))
            if entry.get("ffn_module"):
                ffn = layer._modules[entry["ffn_module"]]
                self._wrap(ffn, f"target/L{index}/ffn_{entry['ffn']}")
                for child, kind in (
                    ("gate", "moe_router"),
                    ("experts", "moe_experts"),
                    ("shared_expert", "moe_shared_expert"),
                    ("shared_experts", "moe_shared_expert"),
                ):
                    module = ffn._modules.get(child)
                    if module is not None and entry["ffn"] != taxonomy.FFN_DENSE:
                        self._wrap(module, f"target/L{index}/{kind}")
        head = getattr(target, "lm_head", None)
        if head is not None:
            self._wrap(head, "target/lm_head")

    def _wrap(self, module, label: str) -> None:
        key = id(module)

        def pre(_module, _args):
            if not self._active():
                return
            scope = torch.autograd.profiler.record_function(MODULE_PREFIX + label)
            scope.__enter__()
            self._stacks[key].append(scope)

        def post(_module, _args, _output):
            stack = self._stacks[key]
            if stack:
                stack.pop().__exit__(None, None, None)

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class AttentionProbe:
    """What each full-attention call was handed: query width and mask.

    SDPA picks its kernel from the inputs. A materialised boolean or additive
    mask rules out the flash kernel and a query wider than one token with no
    mask needs ``is_causal``; which of those happened at q=1 (AR) and at
    q=block (verify) is the fact the verify-cost question turns on, and the
    kernel names in the trace confirm which backend actually ran.
    """

    def __init__(self, phases: PhaseRanges) -> None:
        self._phases = phases
        self.seen: dict[tuple, dict] = {}

    def hook(self, layer_index: int):
        def pre(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            mask = kwargs.get("attention_mask")
            if mask is None and len(args) > 2:
                mask = args[2]
            q = int(hidden.shape[1]) if isinstance(hidden, torch.Tensor) else None
            key = (self._phases.current, q)
            if key in self.seen:
                self.seen[key]["calls"] += 1
                return None
            self.seen[key] = {
                "phase": self._phases.current,
                "query_tokens": q,
                "first_layer": layer_index,
                "mask": (
                    None if mask is None else {
                        "shape": list(mask.shape), "dtype": str(mask.dtype),
                    }
                    if isinstance(mask, torch.Tensor) else type(mask).__name__
                ),
                "attn_implementation": getattr(
                    getattr(module, "config", None), "_attn_implementation", None
                ),
                "is_causal_attr": getattr(module, "is_causal", None),
                "calls": 1,
            }
            return None

        return pre

    def summary(self) -> list[dict]:
        return list(self.seen.values())


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------

def _load_trace(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def _innermost(ranges: list[tuple], starts: list[float], ts: float,
               max_walk: int = 256):
    """The innermost range containing ``ts``; ranges sorted by start."""
    index = bisect.bisect_right(starts, ts) - 1
    walked = 0
    while index >= 0 and walked < max_walk:
        start, end, name = ranges[index]
        if start <= ts <= end:
            return name
        index -= 1
        walked += 1
    return None


def _union(intervals: list[tuple[float, float]]) -> float:
    return events_module.union_seconds(intervals)


def _clip(intervals, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi]


def analyse(path: str, *, phase_counts: dict | None = None,
            top_kernels: int = 25) -> dict:
    """Device busy/idle and per-phase/module/layer device time from a trace."""
    trace = _load_trace(path)
    launches: dict = {}
    phase_ranges: dict = defaultdict(list)
    module_ranges: dict = defaultdict(list)
    gpu: list[dict] = []
    for event in trace:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        args = event.get("args") or {}
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        if category in ("cuda_runtime", "cuda_driver"):
            correlation = args.get("correlation")
            if correlation is not None:
                launches[correlation] = (event.get("tid"), ts)
        elif category == "user_annotation":
            name = event.get("name", "")
            if name.startswith(PHASE_PREFIX):
                phase_ranges[event.get("tid")].append(
                    (ts, ts + dur, name[len(PHASE_PREFIX):])
                )
            elif name.startswith(MODULE_PREFIX):
                module_ranges[event.get("tid")].append(
                    (ts, ts + dur, name[len(MODULE_PREFIX):])
                )
        elif category in ("kernel", "gpu_memcpy", "gpu_memset"):
            # Kernels carry "device"; memcpy/memset carry "inDevice" (the
            # device whose engine ran the copy) plus fromDevice/toDevice.
            device = args.get("device", args.get("inDevice"))
            if device is None:
                continue
            gpu.append({
                "device": int(device),
                "start": ts,
                "end": ts + dur,
                "kind": category,
                "name": event.get("name", ""),
                "bytes": args.get("bytes"),
                "correlation": args.get("correlation"),
            })

    for table in (phase_ranges, module_ranges):
        for tid in table:
            table[tid].sort()
    phase_starts = {tid: [r[0] for r in rows] for tid, rows in phase_ranges.items()}
    module_starts = {tid: [r[0] for r in rows] for tid, rows in module_ranges.items()}

    for event in gpu:
        launch = launches.get(event["correlation"])
        event["phase"] = event["module"] = None
        if launch is None:
            continue
        tid, ts = launch
        if tid in phase_ranges:
            event["phase"] = _innermost(phase_ranges[tid], phase_starts[tid], ts)
        if tid in module_ranges:
            event["module"] = _innermost(module_ranges[tid], module_starts[tid], ts)

    decode = [e for e in gpu if (e["phase"] or "").startswith("decode:")]
    if not decode:
        return {"ok": False, "na_reason": "no device activity attributed to a decode phase"}
    lo = min(e["start"] for e in decode)
    hi = max(e["end"] for e in decode)
    window_s = (hi - lo) * 1e-6
    devices = sorted({e["device"] for e in gpu})

    def seconds(intervals):
        return _union([(a * 1e-6, b * 1e-6) for a, b in intervals])

    all_busy = seconds(_clip([(e["start"], e["end"]) for e in gpu], lo, hi))
    idle_all = max(window_s - all_busy, 0.0)
    per_device = {}
    for device in devices:
        mine = [e for e in gpu if e["device"] == device]
        busy = seconds(_clip([(e["start"], e["end"]) for e in mine], lo, hi))
        kernels = seconds(_clip(
            [(e["start"], e["end"]) for e in mine if e["kind"] == "kernel"], lo, hi
        ))
        copies = [e for e in mine if e["kind"] == "gpu_memcpy" and lo <= e["start"] <= hi]
        p2p = [e for e in copies if "PtoP" in e["name"] or "Peer" in e["name"]]
        idle = max(window_s - busy, 0.0)
        per_device[f"cuda:{device}"] = {
            "busy_s": busy,
            "busy_fraction": busy / window_s if window_s else None,
            "kernel_busy_s": kernels,
            "memcpy_s": seconds([(e["start"], e["end"]) for e in copies]),
            "memcpy_count": len(copies),
            "p2p_memcpy_s": seconds([(e["start"], e["end"]) for e in p2p]),
            "p2p_memcpy_bytes": sum(int(e["bytes"] or 0) for e in p2p),
            "kernel_count": sum(
                1 for e in mine if e["kind"] == "kernel" and lo <= e["start"] <= hi
            ),
            "idle_s": idle,
            "idle_shard_wait_s": max(idle - idle_all, 0.0),
            "idle_all_devices_s": idle_all,
        }

    # Device time by phase / module kind / layer, decode window only. Kernels
    # on one device's stream serialise, so within a (phase, device) cell a sum
    # is a fair stand-in for the union and keeps the cells additive.
    counts = dict(phase_counts or {})
    phases: dict = {}
    for event in decode:
        phase = event["phase"]
        entry = phases.setdefault(phase, {
            "device_s": defaultdict(float),
            "kind_s": defaultdict(float),
            "module_kind_s": defaultdict(float),
            "layer_s": defaultdict(float),
            "kernel_names": defaultdict(lambda: [0.0, 0]),
            "kernels": 0,
            "memcpy_s": 0.0,
        })
        dur = (event["end"] - event["start"]) * 1e-6
        entry["device_s"][f"cuda:{event['device']}"] += dur
        entry["kind_s"][event["kind"]] += dur
        if event["kind"] == "gpu_memcpy":
            entry["memcpy_s"] += dur
        if event["kind"] == "kernel":
            entry["kernels"] += 1
            record = entry["kernel_names"][event["name"]]
            record[0] += dur
            record[1] += 1
        module = event["module"]
        if module:
            parts = module.split("/")
            kind = parts[-1]
            entry["module_kind_s"][kind] += dur
            if len(parts) >= 3 and parts[1].startswith("L"):
                entry["layer_s"][int(parts[1][1:])] += dur
        else:
            entry["module_kind_s"]["outside_target_modules"] += dur

    phase_out = {}
    for phase, entry in phases.items():
        occurrences = counts.get(phase)
        names = sorted(
            entry["kernel_names"].items(), key=lambda item: -item[1][0]
        )[:top_kernels]
        by_category: dict = defaultdict(float)
        for name, (total, _count) in entry["kernel_names"].items():
            by_category[kernel_category(name)] += total
        device_total = sum(entry["device_s"].values())
        phase_out[phase] = {
            "occurrences": occurrences,
            "device_time_s": device_total,
            "device_time_per_occurrence_s": (
                device_total / occurrences if occurrences else None
            ),
            "device_s": dict(entry["device_s"]),
            "memcpy_s": entry["memcpy_s"],
            "kernels": entry["kernels"],
            "kernels_per_occurrence": (
                entry["kernels"] / occurrences if occurrences else None
            ),
            "module_kind_s": dict(entry["module_kind_s"]),
            "layer_s": {str(k): v for k, v in sorted(entry["layer_s"].items())},
            "kernel_category_s": dict(by_category),
            "top_kernels": [
                {"name": name[:200], "total_s": total, "count": count,
                 "category": kernel_category(name)}
                for name, (total, count) in names
            ],
        }

    return {
        "ok": True,
        "decode_window_s": window_s,
        "all_devices_busy_s": all_busy,
        "idle_all_devices_s": idle_all,
        "idle_all_devices_fraction": idle_all / window_s if window_s else None,
        "devices": per_device,
        "phases": phase_out,
        "unattributed_gpu_events": sum(1 for e in gpu if e["phase"] is None),
        "notes": (
            "Window = first to last device event launched from a decode phase. "
            "busy is a union over kernels, memcpy and memset. idle_shard_wait = "
            "idle while another device was busy; idle_all_devices = no device "
            "busy (host launch/sync). Profiling inflates host gaps, so "
            "idle_all_devices is an upper bound for the unprofiled run."
        ),
    }


# ---------------------------------------------------------------------------
# Running a condition under the profiler
# ---------------------------------------------------------------------------

def profile_call(fn, *, target, trace_path: str, step_phase: str,
                 max_steps: int = 32) -> dict:
    """Run ``fn(phase_callback)`` once, profiling a bounded decode window.

    The profiler starts when ``step_phase`` (a verify, or an AR step) opens
    for the first time and stops when it opens for the ``max_steps + 1``-th
    time, or at the end of decode. So the window is ``max_steps`` whole step
    cycles and never contains prefill or the drafter's first (prompt-length)
    call. Profiling a whole run does not scale: on Qwen3.5-35B-A3B the
    reference MoE expert loop launches kernels per expert, and a 4k AR run's
    full trace was 1 GB gzipped and took >100 GB of host memory to parse.
    """
    from torch.profiler import ProfilerActivity, profile

    global PROFILER_USED
    PROFILER_USED = True
    phases = PhaseRanges()
    probe = AttentionProbe(phases)
    modules = ModuleRanges(
        target, attention_probe=probe, active=lambda: phases.recording
    )
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    )
    state = {"stage": "before"}

    def sync_all():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

    def callback(label, site):
        if state["stage"] == "before" and label == step_phase:
            sync_all()
            prof.start()
            phases.recording = True
            state["stage"] = "on"
        elif state["stage"] == "on" and (
            label is None
            or (label == step_phase and phases.counts[step_phase] >= max_steps)
        ):
            phases(None, {})
            phases.recording = False
            # Every device, so the last step's kernels on the far shard land
            # inside the trace.
            sync_all()
            prof.stop()
            state["stage"] = "done"
        phases(label, site)

    try:
        result = fn(callback)
        if state["stage"] == "on":
            phases(None, {})
            phases.recording = False
            sync_all()
            prof.stop()
    finally:
        modules.remove()
    if state["stage"] == "before":
        return {
            "result": result, "trace_file": None,
            "analysis": {"ok": False, "na_reason": f"no {step_phase!r} phase ran"},
            "attention_calls": probe.summary(), "phase_counts": {},
        }
    prof.export_chrome_trace(trace_path)
    analysis = analyse(trace_path, phase_counts=dict(phases.counts))
    steps = phases.counts.get(step_phase, 0)
    if analysis.get("ok") and steps:
        analysis["step_phase"] = step_phase
        analysis["profiled_steps"] = steps
        analysis["window_s_per_step"] = analysis["decode_window_s"] / steps
    return {
        "result": result,
        "trace_file": trace_path,
        "trace_bytes": os.path.getsize(trace_path),
        "analysis": analysis,
        "attention_calls": probe.summary(),
        "phase_counts": dict(phases.counts),
    }


def module_breakdown(path: str, phase: str) -> dict:
    """Device time per occurrence of ``phase``, by module kind x kernel category.

    ``analyse`` splits a phase by module kind and, separately, by kernel
    category. This crosses the two, which is what separates an MoE layer's
    expert GEMMs from its dispatch (index/scatter/gather, elementwise), and an
    attention layer's SDPA kernel from the mask construction around it.
    """
    trace = _load_trace(path)
    launches: dict = {}
    phase_ranges: dict = defaultdict(list)
    module_ranges: dict = defaultdict(list)
    kernels = []
    for event in trace:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        args = event.get("args") or {}
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        if category in ("cuda_runtime", "cuda_driver") and args.get("correlation") is not None:
            launches[args["correlation"]] = (event.get("tid"), ts)
        elif category == "user_annotation":
            name = event.get("name", "")
            if name.startswith(PHASE_PREFIX):
                phase_ranges[event.get("tid")].append((ts, ts + dur, name[len(PHASE_PREFIX):]))
            elif name.startswith(MODULE_PREFIX):
                module_ranges[event.get("tid")].append((ts, ts + dur, name[len(MODULE_PREFIX):]))
        elif category == "kernel":
            kernels.append((args.get("correlation"), dur, event.get("name", "")))
    for table in (phase_ranges, module_ranges):
        for tid in table:
            table[tid].sort()
    phase_starts = {tid: [r[0] for r in rows] for tid, rows in phase_ranges.items()}
    module_starts = {tid: [r[0] for r in rows] for tid, rows in module_ranges.items()}
    occurrences = sum(1 for rows in phase_ranges.values() for r in rows if r[2] == phase)
    if not occurrences:
        return {"ok": False, "na_reason": f"no {phase!r} ranges in trace"}
    seconds: dict = defaultdict(lambda: defaultdict(float))
    counts: dict = defaultdict(lambda: defaultdict(int))
    for correlation, dur, name in kernels:
        launch = launches.get(correlation)
        if launch is None:
            continue
        tid, ts = launch
        if tid not in phase_ranges or _innermost(phase_ranges[tid], phase_starts[tid], ts) != phase:
            continue
        module = (
            _innermost(module_ranges[tid], module_starts[tid], ts)
            if tid in module_ranges else None
        )
        kind = module.split("/")[-1] if module else "outside_target_modules"
        category = kernel_category(name)
        seconds[kind][category] += dur * 1e-6
        counts[kind][category] += 1
    return {
        "ok": True,
        "trace_file": path,
        "phase": phase,
        "occurrences": occurrences,
        "per_occurrence_ms": {
            kind: {cat: value / occurrences * 1e3 for cat, value in cats.items()}
            for kind, cats in seconds.items()
        },
        "kernels_per_occurrence": {
            kind: {cat: value / occurrences for cat, value in cats.items()}
            for kind, cats in counts.items()
        },
    }
