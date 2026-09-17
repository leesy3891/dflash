"""Batch-size sweep at a fixed context length. See BATCH_PROFILING_PLAN.md.

One process takes one ``--context-length`` and walks ``--batch-sizes``. Every
batch size processes the same prompt pool in the same seeded order, split into
``pool / B`` batches, so the points differ in batching and nothing else. Each
(B) point writes one record holding both modes.
"""

from __future__ import annotations

import copy
import gc
import json
import random
import statistics
from datetime import datetime
from pathlib import Path

_GB = float(1 << 30)


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _sum(values):
    values = [v for v in values if v is not None]
    return sum(values) if values else None


def _gdn_kernels() -> dict:
    """Which implementation transformers bound for the Qwen3.5 GDN kernels.

    The binding happens once, when ``modeling_qwen3_5`` is imported: if ``fla``
    resolves, the decorated name calls it, otherwise the torch fallback stays.
    Nothing announces the choice, and it changes both the verify and the
    rollback replay, so the record has to carry it or two sweeps measured with
    different kernels look alike.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
    except ImportError:
        return {}
    from dflash import batch as batch_module
    bound = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        function = getattr(qwen3_5, name, None)
        if getattr(function, "__module__", None) == batch_module.__name__:
            # The engine's capture has already replaced the module attribute;
            # what it wrapped is the binding this run actually calls.
            function = batch_module._CHUNK
        modules = [cell.cell_contents.__module__ for cell in (getattr(function, "__closure__", None) or ())
                   if callable(getattr(cell, "cell_contents", None))]
        bound[name] = modules[0] if modules else None
    try:
        import fla
        bound["fla_version"] = fla.__version__
    except ImportError:
        bound["fla_version"] = None
    return bound


def _acceptance_split(batch: dict) -> tuple[list[int], list[int]]:
    """Per-verify kept counts before and after each row's natural EOS."""
    before, after = [], []
    for kept, eos in zip(batch["kept"], batch["natural_eos_at"]):
        position = 1
        for value in kept:
            (after if eos is not None and position > eos else before).append(value)
            position += value
    return before, after


def summarize(batches: list[dict], num_tokens: int, block_size: int,
              target_weight_bytes: int) -> dict:
    from . import record as record_module

    rows = sum(len(b["tokens"]) for b in batches)
    prefill = sum(sum(b["prefill_s"]) for b in batches)
    decode = sum(b["decode_s"] for b in batches)
    decode_tokens = rows * (num_tokens - 1)

    # Full occupancy: the steps before any row of the batch has finished, so
    # every one of them ran at the nominal batch size.
    full_tokens = full_wall = 0.0
    for batch in batches:
        width = len(batch["tokens"])
        for step in batch["steps"]:
            if step["active"] < width:
                break
            full_tokens += step["tokens"]
            full_wall += step["wall_s"]

    steps = [s for b in batches for s in b["steps"]]
    parts = {name: _sum(s[f"{name}_s"] for s in steps) for name in ("verify", "accept", "ctx", "draft")}
    gpu = sum(v for v in parts.values() if v is not None)

    per_request_tpot, queue_wait = [], []
    for batch in batches:
        walls = [s["wall_s"] for s in batch["steps"]]
        for row, finished in enumerate(batch["finished_at_step"]):
            if finished is not None and num_tokens > 1:
                per_request_tpot.append(sum(walls[: finished + 1]) / (num_tokens - 1))
            queue_wait.append(sum(batch["prefill_s"][:row]))

    kept = [k for b in batches for row in b["kept"] for k in row]
    before, after = [], []
    for batch in batches:
        b, a = _acceptance_split(batch)
        before += b
        after += a

    runs = [{"phase_memory": b["phase_memory"]} for b in batches]
    phases = record_module.aggregate_phases(runs, target_weight_bytes)
    peak = max(b["peak_bytes"] for b in batches)
    summary = {
        "rows": rows,
        "num_batches": len(batches),
        "output_tokens_per_row": num_tokens,
        # Throughput
        "decode_tok_s_makespan": decode_tokens / decode if decode else None,
        "decode_tok_s_full_occupancy": full_tokens / full_wall if full_wall else None,
        "full_occupancy_fraction_of_decode": full_wall / decode if decode else None,
        "e2e_tok_s": rows * num_tokens / (prefill + decode),
        "total_prefill_s": prefill,
        "total_decode_s": decode,
        "mean_ttft_s": _mean([t for b in batches for t in b["prefill_s"]]),
        "mean_queue_wait_s": _mean(queue_wait),
        "mean_request_tpot_s": _mean(per_request_tpot),
        # Decode step decomposition (GPU time from CUDA events; cpu is the rest)
        "decode_steps": len(steps),
        "mean_step_wall_s": decode / len(steps) if steps else None,
        "mean_tokens_per_step": _mean([s["tokens"] for s in steps]),
        "step_time_s": {**{k: (v / len(steps) if v is not None else None) for k, v in parts.items()},
                        "cpu_and_sync": (decode - gpu) / len(steps) if steps else None},
        "decode_time_share": {**{k: (v / decode if v is not None else None) for k, v in parts.items()},
                              "cpu_and_sync": (decode - gpu) / decode if decode else None},
        # Memory
        "peak_memory_gb": peak / _GB,
        "reserved_after_gb": max(b["reserved_bytes"] for b in batches) / _GB,
        "target_kv_alloc_gb": max(b["target_kv_alloc_bytes"] for b in batches) / _GB,
        "draft_kv_alloc_gb": max(b["draft_kv_alloc_bytes"] for b in batches) / _GB,
        "gdn_state_gb": max(b["gdn_state_bytes"] for b in batches) / _GB,
        "target_weight_gb": target_weight_bytes / _GB,
        "draft_weight_gb": batches[0]["draft_weight_bytes"] / _GB,
        "phase_memory": phases,
        "peak_phase": record_module.peak_phase(phases),
        "peak_site": max(batches, key=lambda b: b["peak_bytes"])["peak_site"],
        "natural_eos_rows": sum(e is not None for b in batches for e in b["natural_eos_at"]),
    }
    if block_size > 1:
        stage_first: dict = {}
        stage_steady: dict = {}
        for batch in batches:
            for bucket, into in (("first", stage_first), ("steady", stage_steady)):
                for name, value in ((batch["stage_s"] or {}).get(bucket) or {}).items():
                    if value is not None:
                        into[name] = into.get(name, 0.0) + value
        steady_calls = sum(b["steady_draft_calls"] for b in batches)
        summary.update({
            "mean_acceptance_length": _mean(kept),
            "mean_acceptance_length_pre_eos": _mean(before),
            "mean_acceptance_length_post_eos": _mean(after),
            "acceptance_length_histogram": [kept.count(n) / len(kept) for n in range(block_size + 1)] if kept else [],
            "mean_drafter_prefill_s": _mean([t for b in batches for t in b["drafter_prefill_s"]]),
            "steady_draft_calls": steady_calls,
            "mean_steady_draft_call_s": (parts["draft"] / steady_calls) if steady_calls and parts["draft"] else None,
            "first_draft_stage_s_per_row": {k: v / rows for k, v in stage_first.items()} or None,
            "steady_draft_stage_s_per_call": {k: v / steady_calls for k, v in stage_steady.items()} if steady_calls else None,
        })
    return summary


def _batch_trace(batch: dict, indices: list[int]) -> dict:
    keep = ("prompt_lengths", "capacity", "prefill_s", "drafter_prefill_s", "decode_s",
            "steps", "finished_at_step", "steady_draft_calls", "peak_bytes", "peak_site",
            "target_kv_alloc_bytes", "draft_kv_alloc_bytes", "gdn_state_bytes")
    return {"prompts": indices, **{k: batch[k] for k in keep}}


def run_batch_sweep(args) -> None:
    import torch
    import transformers

    from . import record as record_module
    from .batch import BatchEngine
    from .benchmark import _build_context_samples, load_transformers_models, stop_token_ids

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    block_size = args.block_size if args.block_size is not None else draft.block_size
    # Every prompt exactly --context-length tokens: the fitter lands within ±16
    # (in practice ±1) on a few, and equal lengths are what let a batch whose
    # rows advance in lockstep (the baseline) use the dense attention kernel.
    wanted = args.max_samples if args.max_samples is not None else 32
    widened = copy.copy(args)
    widened.max_samples = wanted + 16
    samples, tasks, task_report = _build_context_samples(widened, tokenizer)
    exact = [s for s in samples if s["num_input_tokens"] == args.context_length]
    samples = (exact + [s for s in samples if s not in exact])[:wanted]
    prompts = [
        tokenizer.encode(s["prompt"], return_tensors="pt", add_special_tokens=False)[0].to(device)
        for s in samples
    ]
    print(f"[prompts] {sum(int(p.numel()) == args.context_length for p in prompts)}/{len(prompts)} "
          f"exactly {args.context_length} tokens", flush=True)
    order = list(range(len(prompts)))
    random.Random(args.batch_seed).shuffle(order)
    engine = BatchEngine(
        target, draft, eos_ids=stop_token_ids(target, tokenizer),
        eos_mode=args.eos_mode, draft_stages=args.profile_draft_stages,
    )
    modes = {"baseline": 1, "dflash": block_size} if args.baseline else {"dflash": block_size}
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    num_tokens = args.fixed_output_tokens

    # Full-length prefill kernels once, then each batch shape before its point.
    for size in modes.values():
        engine.run_batch([prompts[order[0]]], 8, size)
    torch.cuda.synchronize()

    failed: dict[str, int] = {}
    out_dir = Path(args.record_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for batch_size in sizes:
        groups = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        status, oom, summaries, traces, outputs = {}, {}, {}, {}, {}
        for mode, size in modes.items():
            if mode in failed:
                status[mode] = f"skipped: OOM at B={failed[mode]}"
                continue
            try:
                engine.run_batch([prompts[i][:512] for i in groups[0]], 4, size)
                batches = []
                for group in groups:
                    batches.append(engine.run_batch([prompts[i] for i in group], num_tokens, size))
                    print(
                        f"[B={batch_size} {mode}] batch {len(batches)}/{len(groups)} "
                        f"decode {batches[-1]['decode_s']:.1f}s peak {batches[-1]['peak_bytes'] / _GB:.2f} GB",
                        flush=True,
                    )
                status[mode] = "ok"
                summaries[mode] = summarize(batches, num_tokens, size, engine.target_weight_bytes)
                summaries[mode]["attention_kernels"] = batches[-1]["attention_kernels"]
                traces[mode] = [_batch_trace(b, g) for b, g in zip(batches, groups)]
                outputs[mode] = {
                    i: {"tokens": row, "kept": kept, "natural_eos_at": eos}
                    for b, g in zip(batches, groups)
                    for i, row, kept, eos in zip(g, b["tokens"], b["kept"], b["natural_eos_at"])
                }
            except torch.OutOfMemoryError as exc:
                status[mode] = "oom"
                failed[mode] = batch_size
                oom[mode] = {"phase": getattr(exc, "dflash_phase", None),
                             "message": str(exc).splitlines()[0][:300]}
                print(f"[B={batch_size} {mode}] OOM in {oom[mode]['phase']}", flush=True)
            finally:
                batches = None
                gc.collect()
                torch.cuda.empty_cache()

        speedup = None
        if len(summaries) == 2:
            d, b = summaries["dflash"], summaries["baseline"]
            speedup = {
                "same_b_makespan": d["decode_tok_s_makespan"] / b["decode_tok_s_makespan"],
                "same_b_full_occupancy": d["decode_tok_s_full_occupancy"] / b["decode_tok_s_full_occupancy"],
                "same_b_e2e": d["e2e_tok_s"] / b["e2e_tok_s"],
            }
        rows = []
        for index in order:
            entry = {"prompt": index, "task": samples[index]["task"],
                     "input_tokens": int(prompts[index].numel())}
            for mode in outputs:
                entry[mode] = outputs[mode][index]
            if len(outputs) == 2:
                entry["dflash_matches_baseline"] = (
                    outputs["dflash"][index]["tokens"] == outputs["baseline"][index]["tokens"]
                )
            rows.append(entry)
        payload = {
            "git_commit": record_module._git_commit(),
            "backend": "transformers-batched",
            "model": args.model, "draft": args.draft, "model_name": args.model_name,
            "context_length": args.context_length,
            "context_tasks": tasks, "context_task_report": task_report,
            "batch_size": batch_size, "num_prompts": len(prompts), "batch_seed": args.batch_seed,
            "fixed_output_tokens": num_tokens, "eos_mode": args.eos_mode,
            "block_size": block_size, "reasoning": args.reasoning,
            "hidden_states": "selective", "prefill_mode": "sequential", "kv_cache": "static",
            "gdn_rollback": "replay", "gdn_prefill_record": False,
            "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gdn_kernels": _gdn_kernels(),
            "status": status, "oom": oom, "summary": summaries, "speedup": speedup,
            "batches": traces, "rows": rows,
        }
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"{args.model_name}_{args.context_length}_b{batch_size}_{stamp}.json"
        path.write_text(json.dumps(payload, indent=1))
        line = " ".join(
            f"{m}: {s['decode_tok_s_makespan']:.1f} tok/s peak {s['peak_memory_gb']:.2f} GB"
            for m, s in summaries.items()
        )
        extra = f" speedup {speedup['same_b_makespan']:.2f}x" if speedup else ""
        print(f"[B={batch_size}] {status} {line}{extra} -> {path}", flush=True)
