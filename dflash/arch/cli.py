"""``python -m dflash.arch`` -- the entry points for the main-branch B=1 sweep.

Each subcommand is one pass, run on its own, because they interfere with each
other and because a long pass that fails should not take a short one with it.
Everything writes under ``record_arch_main/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import env, models, sweep

RECORD_ROOT = sweep.RECORD_ROOT


def _write(path: str, payload) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=str)
    return path


def cmd_manifest(args) -> None:
    payload = env.manifest({"device_health": env.device_health()})
    path = _write(os.path.join(args.record_root, "manifest.json"), payload)
    print(json.dumps(
        {
            "git": payload["git"]["commit"],
            "dirty": payload["git"]["dirty"],
            "device_health": payload["device_health"],
            "versions": payload["versions"],
        },
        indent=1,
    ))
    print(f"manifest -> {path}")


def cmd_plan(args) -> None:
    payload = sweep.plan(args.models, repeats=args.repeats)
    path = _write(os.path.join(args.record_root, "plan.json"), payload)
    for key, entry in payload["models"].items():
        if "error" in entry:
            print(f"{key}: ERROR {entry['error']}")
            continue
        print(f"== {key}  compatible={entry['pairing']['compatible']}")
        for condition in entry["conditions"]:
            flags = ",".join(condition["flags"]) or "-"
            print(f"   {condition['key']:<52} {condition['sweep']:<14} {flags}")
    print(f"plan -> {path}")


def cmd_taxonomy(args) -> None:
    from . import loader, taxonomy

    pair = models.resolve(args.model)
    devices = _devices(args)
    target, report = loader.load_target(
        pair.target, devices=devices, shard=len(devices) > 1
    )
    spec = report["taxonomy"]
    analytic = taxonomy.per_token_state_bytes(spec)
    payload = {"model": args.model, **report["target"], "analytic_state": analytic,
               "layers": spec["layers"]}
    path = _write(
        os.path.join(args.record_root, args.model, "taxonomy.json"), payload
    )
    print(f"{args.model}: {spec['num_layers']} layers "
          f"mixers={spec['mixer_counts']} ffn={spec['ffn_counts']}")
    print(f"  layers/device        : {spec['layers_per_device']}")
    print(f"  weights/device bytes : {spec['param_bytes_per_device']}")
    print(f"  attention KV/token   : {analytic['attention_kv_bytes_per_token']:,} B")
    print(f"  GDN state (fixed)    : {analytic['gdn_state_bytes_total']:,} B")
    print(f"taxonomy -> {path}")


def cmd_rollback(args) -> None:
    from . import loader, rollback

    pair = models.resolve(args.model)
    devices = _devices(args)
    target, report = loader.load_target(
        pair.target, devices=devices, shard=len(devices) > 1
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pair.target)
    text = "The history of computing is a history of abstractions. " * 2000
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    ids = ids[:, : args.prompt_tokens]

    results = []
    for accepted in args.accepted:
        result = rollback.check(
            target, ids, block_size=args.block_size,
            accepted=accepted, probe_tokens=args.probe_tokens,
        )
        print(rollback.summarise(result), flush=True)
        results.append(result)
    payload = {
        "model": args.model,
        "manifest": env.manifest(),
        "target": report["target"],
        "prompt_tokens": args.prompt_tokens,
        "results": results,
    }
    path = _write(
        os.path.join(args.record_root, args.model, "rollback.json"), payload
    )
    print(f"rollback -> {path}")


def cmd_verifyq(args) -> None:
    from transformers import AutoTokenizer

    from . import loader, prompts, verifyq

    pair = models.resolve(args.model)
    devices = _devices(args)
    target, report = loader.load_target(
        pair.target, devices=devices, shard=len(devices) > 1
    )
    tokenizer = AutoTokenizer.from_pretrained(pair.target)
    # The sweep's own prompts, not filler: acceptance does not enter here, but
    # MoE routing does, and routing follows content.
    prompt_sets = prompts.build_set(
        tokenizer, args.input_tokens, mode=prompts.MODE_FIXED_TASK,
        task=args.prompt_task, per_length=1, seed=args.prompt_seed,
    )

    results = []
    for length in args.input_tokens:
        prompt = prompt_sets[length][0]
        ids = prompt["input_ids"]
        block = None
        for policy in args.cache_policies:
            result = verifyq.measure(
                target, ids, widths=tuple(args.widths), repeats=args.repeats,
                cache_policy_recording=policy == "recording", block_ids=block,
            )
            # Every policy verifies the same tokens.
            block = result.pop("block_ids_tensor")
            result["prompt"] = _prompt_summary(prompt)
            results.append({"ok": True, **result})
            print(f"S={length} policy={policy} block={result['block_source']}")
            for row in result["results"]:
                print(
                    f"   q={row['q']:<3} median={row['median_s']*1e3:8.3f} ms "
                    f"stdev={row['stdev_s']*1e3:6.3f} "
                    f"x_over_q1={row['cost_over_q1']:.3f}"
                )
            print(f"   snapshot={result['snapshot_s']*1e3:.2f} ms "
                  f"restore={result['mean_restore_s']*1e3:.2f} ms "
                  f"({result['snapshot_bytes']/2**20:.1f} MiB)")

    payload = {
        "model": args.model, "manifest": env.manifest(),
        "gpu_selection": args.gpu_selection,
        "target": report["target"], "results": results,
    }
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = _write(
        os.path.join(
            args.record_root, args.model,
            f"verify_width_{len(devices)}gpu_{stamp}.json",
        ),
        payload,
    )
    print(f"verify width -> {path}")


def _print_lossless(block: dict) -> None:
    def agree(name):
        value = block.get(name)
        if not value:
            return "n/a"
        return ("identical" if value["identical"]
                else f"diverge@{value['first_divergence']}")
    print(f"  lossless: stock_vs_ar={agree('stock_vs_ar')} "
          f"exact_vs_ar={agree('exact_vs_ar')} "
          f"audit_follows_perf={block.get('audit_follows_perf')} "
          f"accept stock={block.get('acceptance_rate_stock')} "
          f"exact={block.get('acceptance_rate_exact')}", flush=True)
    audit = block.get("audit") or {}
    accumulated = audit.get("accumulated") or {}
    rejecting = (accumulated.get("rejecting") or {}).get("state_rel_err_mean") or {}
    if accumulated:
        print(f"            state drift (accum, rejecting steps) "
              f"median={rejecting.get('median')} max={rejecting.get('max')} "
              f"final={accumulated.get('final_state_rel_err_mean')} "
              f"argmax flips {accumulated.get('argmax_flips')}/"
              f"{accumulated.get('next_token_steps')}", flush=True)


def _devices(args) -> list[int]:
    """The pinned devices, checked now that CUDA is up."""
    selection = env.check_pinned(args.gpu_selection)
    args.gpu_selection = selection
    return selection["devices"]


def _prompt_summary(prompt: dict) -> dict:
    return {k: v for k, v in prompt.items() if k not in ("input_ids", "prompt")}


def _safe(key: str) -> str:
    return key.replace("/", "_")


def cmd_sweep(args) -> None:
    """Run one model's conditions: every AR run first, then the drafter.

    The drafter is loaded only after the last AR run, so the AR peaks are a
    drafter-free baseline by construction rather than by assumption, and the
    record says ``drafter_resident: false`` on every AR row.
    """
    import dataclasses
    from datetime import datetime, timezone

    import torch
    from transformers import AutoConfig, AutoTokenizer

    from . import backend, loader, prompts, report, run
    from .run import PASS_MEMORY, PASS_MOE, PASS_PERF, PASS_TRACE

    devices = _devices(args)
    pair = models.resolve(args.model)
    target_config = AutoConfig.from_pretrained(pair.target)
    draft_config = AutoConfig.from_pretrained(pair.draft)
    pairing = models.check_pairing(target_config, draft_config)
    if not pairing["compatible"]:
        raise ValueError(
            f"drafter {pair.draft} is not compatible with target {pair.target}: "
            f"{pairing['problems']}"
        )
    target, target_load = loader.load_target(
        pair.target, devices=devices, shard=len(devices) > 1
    )
    tokenizer = AutoTokenizer.from_pretrained(pair.target)
    target_spec = target_load["taxonomy"]
    target_mixers = [layer["mixer"] for layer in target_spec["layers"]]

    from ..benchmark import stop_token_ids
    stop = stop_token_ids(target, tokenizer)
    limits = sweep.context_limits(pair, target.config, draft_config)

    base_conditions = sweep.conditions(
        args.model,
        sweeps=tuple(args.sweeps),
        repeats=args.repeats,
        natural_output=not args.no_natural,
    )
    if args.input_tokens:
        base_conditions = [
            c for c in base_conditions if c.input_tokens in set(args.input_tokens)
        ]
    prompt_sets = prompts.build_set(
        tokenizer,
        sorted({c.input_tokens for c in base_conditions}),
        mode=args.prompt_mode,
        task=args.prompt_task,
        per_length=args.prompts_per_length,
        seed=args.prompt_seed,
    )
    legacy = args.prompt_mode == prompts.MODE_LEGACY
    work = []
    for condition in base_conditions:
        for prompt in prompt_sets[condition.input_tokens]:
            work.append((
                dataclasses.replace(
                    condition,
                    prompt_index=None if legacy else prompt["prompt_index"],
                ),
                prompt,
            ))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    trace_dir = os.path.join(args.record_root, args.model, "traces", stamp)
    first_device = f"cuda:{devices[0]}"

    # Order matters. Once torch.profiler has run in a process, every later
    # kernel launch is slower (measured: AR TPOT +13% on Qwen3-8B, +24% on
    # Qwen3.5-35B-A3B, persisting for the life of the process), so every
    # timing pass runs before any diagnostic pass touches the profiler.
    #   A. AR timing, target only (drafter-free memory baseline)
    #   B. DFlash timing/memory/lossless, drafter loaded
    #   C. diagnostics: moe routing, then traces (AR and DFlash)
    timing_passes = [p for p in args.passes if p not in (PASS_TRACE, PASS_MOE)]

    # ---- Phase A: AR timing, target only -------------------------------------
    entries = []
    for number, (condition, prompt) in enumerate(work):
        print(f"\n=== [AR] {condition.key()}", flush=True)
        input_ids = prompt["input_ids"].to(first_device)
        entry = {
            **condition.as_dict(),
            "key": condition.key(),
            **sweep.flag_condition(condition, limits),
            "prompt": _prompt_summary(prompt),
            "passes": {},
            "ar": {},
        }
        policies = list(args.ar_policies)
        if args.ar_order == "alternate" and number % 2:
            policies.reverse()
        entry["ar_policy_order"] = policies
        for policy in policies:
            result = run.run_ar(
                target, input_ids, condition, stop_token_ids=stop,
                cache_policy=policy, target_mixers=target_mixers,
                drafter_resident=False,
            )
            entry["ar"][policy] = result.as_dict()
            status = (
                f"tpot={result.median.get('time_per_output_token_s', 0)*1e3:.2f} ms"
                if result.ok else f"FAILED {result.error_class}"
            )
            print(f"  ar/{policy:<10} {status}", flush=True)
        entries.append((condition, prompt, entry))

    # ---- Phase B: DFlash timing, drafter loaded ------------------------------
    run._fresh_allocator()
    draft, draft_load = loader.load_draft(pair.draft, device=devices[0])
    draft_spec = draft_load.get("taxonomy") or {}
    draft_mixers = [layer["mixer"] for layer in draft_spec.get("layers", [])]
    for condition, prompt, entry in entries:
        print(f"\n=== [DFlash] {condition.key()}", flush=True)
        input_ids = prompt["input_ids"].to(first_device)
        for pass_name in timing_passes:
            result = run.run_dflash(
                target, draft, input_ids, condition, stop_token_ids=stop,
                pass_name=pass_name, target_mixers=target_mixers,
                draft_mixers=draft_mixers, hidden_states=args.hidden_states,
            )
            entry["passes"][pass_name] = result.as_dict()
            status = (
                f"tpot={result.median.get('time_per_output_token_s', 0)*1e3:.2f} ms "
                f"accept={result.median.get('acceptance_rate')}"
                if result.ok else f"FAILED {result.error_class}: {result.error}"
            )
            print(f"  dflash/{pass_name:<7} {status}", flush=True)

    # ---- Phase C: diagnostics (sync-bracketed hooks, profiler) ---------------
    diagnostics = [p for p in (PASS_MOE, PASS_TRACE) if p in args.passes]
    for pass_name in diagnostics:
        for condition, prompt, entry in entries:
            print(f"\n=== [{pass_name}] {condition.key()}", flush=True)
            input_ids = prompt["input_ids"].to(first_device)
            if pass_name == PASS_MOE:
                result = run.run_ar_moe(
                    target, input_ids, condition, stop_token_ids=stop,
                )
                entry["ar_moe"] = result.as_dict()
                print(f"  ar/moe       {'ok' if result.ok else result.error_class}",
                      flush=True)
                result = run.run_moe(
                    target, draft, input_ids, condition, stop_token_ids=stop,
                    hidden_states=args.hidden_states,
                )
                entry["passes"][PASS_MOE] = result.as_dict()
                print(f"  dflash/moe   {'ok' if result.ok else result.error_class}",
                      flush=True)
            else:
                # The drafter is resident here; the AR trace measures time,
                # not memory, so that does not matter.
                result = run.run_ar_trace(
                    target, input_ids, condition, stop_token_ids=stop,
                    trace_path=os.path.join(
                        trace_dir, f"{_safe(condition.key())}__ar.json.gz"
                    ),
                    max_steps=args.trace_steps,
                )
                entry["ar_trace"] = result.as_dict()
                _print_trace("ar/trace", result)
                result = run.run_trace(
                    target, draft, input_ids, condition, stop_token_ids=stop,
                    hidden_states=args.hidden_states,
                    trace_path=os.path.join(
                        trace_dir, f"{_safe(condition.key())}__dflash.json.gz"
                    ),
                    warmup=True,
                    max_steps=args.trace_steps,
                )
                entry["passes"][PASS_TRACE] = result.as_dict()
                _print_trace("dflash/trace", result)

    records = []
    for condition, prompt, entry in entries:
        perf = entry["passes"].get(PASS_PERF)
        if perf:
            entry["speedup"] = {
                policy: run.speedup(
                    run.PassResult(**perf), run.PassResult(**value)
                )
                for policy, value in entry["ar"].items()
            }
        if {"audit", "exact"} & set(entry["passes"]):
            entry["lossless"] = run.lossless_comparison(entry["passes"], entry["ar"])
            _print_lossless(entry["lossless"])
        if PASS_PERF in entry["passes"] and PASS_MEMORY in entry["passes"]:
            entry["pass_comparison"] = run.compare_passes(
                run.PassResult(**entry["passes"][PASS_PERF]),
                run.PassResult(**entry["passes"][PASS_MEMORY]),
            )
        entry["trace_perturbation"] = _trace_perturbation(entry)
        records.append(entry)

    gdn_backend = backend.probe()
    payload = {
        "model_key": args.model,
        "manifest": env.manifest({"device_health": env.device_health()}),
        "gpu_selection": args.gpu_selection,
        "backend": backend.describe(target),
        "load": {
            "pair": pair.__dict__.copy(),
            "pairing": pairing,
            "target": target_load["target"],
            "draft_report": {k: v for k, v in draft_load.items() if k != "taxonomy"},
            "drafter_loaded_after_ar": True,
        },
        "target_taxonomy": {
            k: v for k, v in target_spec.items() if k != "layers"
        },
        "limits": limits,
        "hidden_states": args.hidden_states,
        "prompt_set": {
            "mode": args.prompt_mode,
            "task": None if legacy else args.prompt_task,
            "per_length": 1 if legacy else args.prompts_per_length,
            "seed": args.prompt_seed,
            "same_source_across_lengths": all(
                p.get("same_source_across_lengths", False)
                for ps in prompt_sets.values() for p in ps
            ) if not legacy else False,
        },
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "trace_dir": trace_dir if PASS_TRACE in args.passes else None,
        "conditions": records,
    }
    for entry in payload["conditions"]:
        entry["model_key"] = args.model
    payload["gdn_backend"] = {
        k: gdn_backend[k] for k in ("config_label", "homogeneous", "summary")
    }
    path = sweep.write_record(
        args.model, payload, root=args.record_root,
        label=f"sweep_{gdn_backend['config_label']}_{len(devices)}gpu",
        stamp=stamp,
    )
    written = report.export(payload, report.csv_dir_for(path))
    print(f"\nsweep -> {path}")
    for name, csv_path in written.items():
        print(f"  {name:<12} -> {csv_path}")


# What changed in how things are measured, written into every record so a
# reader can tell which definitions a number was taken under.
MEASUREMENT_PROTOCOL = {
    "version": 2,
    "ar_before_drafter_load": True,
    "perf_pass_memory_tracking": False,
    "perf_pass_layer_hooks": False,
    "tpot_denominator": "output_tokens - 1 (both paths)",
    "allocator_reset": "empty_cache + reset_peak_memory_stats once per pass",
    "device_maxima_definition": (
        "peak_memory_device_maxima_bytes = per-device max over the run on both "
        "paths; peak_memory_per_device_at_aggregate_peak_bytes = split at the "
        "simultaneous peak (memory pass only)"
    ),
    "timer_fields": "*_total_s totals, *_mean_s per-call means",
    "draft_logits_timed": True,
    "csv_dir": "csv/<record basename>/",
    "trace_window": "first --trace-steps step cycles of decode",
    "pass_order": "AR timing -> drafter load -> DFlash timing -> moe -> trace",
    "timing_rows_record_profiler_used_before": True,
}


def _trace_perturbation(entry: dict) -> dict:
    """Profiled wall time per step against the unprofiled run's.

    The trace covers only a window of steps, so the comparison is per step:
    window length / profiled steps against perf decode / steps (DFlash) or
    AR native TPOT. Approximate -- per-step cost drifts as the context grows
    and the window is the first steps -- but it sizes the profiler's host
    overhead, which lands in idle_all_devices.
    """
    out = {}
    trace = ((entry["passes"].get("trace") or {}).get("median") or {}).get("analysis") or {}
    perf = (entry["passes"].get("perf") or {}).get("median") or {}
    if trace.get("window_s_per_step") and perf.get("decode_latency_s") and perf.get("num_verify_steps"):
        clean = perf["decode_latency_s"] / perf["num_verify_steps"]
        out["dflash"] = {
            "profiled_s_per_step": trace["window_s_per_step"],
            "perf_s_per_step": clean,
            "ratio": trace["window_s_per_step"] / clean,
        }
    ar_trace = ((entry.get("ar_trace") or {}).get("median") or {}).get("analysis") or {}
    native = ((entry["ar"].get("native") or {}).get("median") or {})
    if ar_trace.get("window_s_per_step") and native.get("time_per_output_token_s"):
        out["ar"] = {
            "profiled_s_per_step": ar_trace["window_s_per_step"],
            "perf_s_per_step": native["time_per_output_token_s"],
            "ratio": ar_trace["window_s_per_step"] / native["time_per_output_token_s"],
        }
    return out


def _print_trace(label: str, result) -> None:
    if not result.ok:
        print(f"  {label:<12} FAILED {result.error_class}: {result.error}", flush=True)
        return
    analysis = result.median.get("analysis") or {}
    if not analysis.get("ok"):
        print(f"  {label:<12} {analysis.get('na_reason')}", flush=True)
        return
    busy = " ".join(
        f"{device}={value['busy_fraction']:.2f}"
        for device, value in analysis["devices"].items()
    )
    print(f"  {label:<12} busy {busy} all-idle="
          f"{analysis['idle_all_devices_fraction']:.2f}", flush=True)


def cmd_trace_breakdown(args) -> None:
    from . import trace

    for path in args.traces:
        result = trace.module_breakdown(path, args.phase)
        slug = args.phase.replace(": ", "_").replace(" ", "_")
        out = args.out or path.replace(".json.gz", f".{slug}.breakdown.json")
        _write(out, result)
        print(f"{path}  [{args.phase}]  occurrences={result.get('occurrences')}")
        for kind, cats in sorted(
            (result.get("per_occurrence_ms") or {}).items(),
            key=lambda item: -sum(item[1].values()),
        ):
            detail = ", ".join(
                f"{cat}={value:.2f}" for cat, value in
                sorted(cats.items(), key=lambda item: -item[1])
            )
            print(f"  {kind:<24} {sum(cats.values()):7.2f} ms  ({detail})")
        print(f"  -> {out}")


def cmd_export(args) -> None:
    """Rebuild a record's CSVs from its JSON, into ``csv/<record name>/``."""
    from . import report

    for path in args.records:
        record = report.load(path)
        out_dir = args.out_dir or report.csv_dir_for(path)
        written = report.export(record, out_dir)
        print(f"{path}")
        for name, csv_path in written.items():
            print(f"  {name:<12} -> {csv_path}")


# Subcommands that touch a GPU, and so need the device count pinned before
# CUDA initialises.
GPU_COMMANDS = {"manifest", "taxonomy", "rollback", "verify-width", "sweep"}


def _single_device(parser) -> None:
    parser.add_argument(
        "--single-device", action="store_true",
        help="deprecated: same as the global --num-gpus 1",
    )


def build_parser() -> argparse.ArgumentParser:
    from .prompts import DEFAULT_TASK, MODE_FIXED_TASK, MODE_LEGACY
    from .run import DFLASH_PASSES

    parser = argparse.ArgumentParser(
        prog="python -m dflash.arch",
        description="B=1 architecture profiling for the main branch",
    )
    parser.add_argument(
        "--record-root", default=RECORD_ROOT,
        help="where records are written (default: record_arch_main)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=2,
        help="exactly this many GPUs, idle ones picked by UUID and pinned "
        "through CUDA_VISIBLE_DEVICES before CUDA starts (default: 2)",
    )
    parser.add_argument(
        "--gpu-uuids", nargs="+", default=None,
        help="use these cards (nvidia-smi UUIDs) instead of picking idle ones",
    )
    parser.add_argument(
        "--allow-busy-gpus", action="store_true",
        help="pick cards even if another process holds memory on them",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser(
        "manifest", help="write the environment/GPU manifest"
    )
    manifest.set_defaults(func=cmd_manifest)

    plan = commands.add_parser(
        "plan", help="print the conditions a sweep would run; loads no weights"
    )
    plan.add_argument("models", nargs="*", default=sorted(models.REGISTRY))
    plan.add_argument("--repeats", type=int, default=3)
    plan.set_defaults(func=cmd_plan)

    taxonomy = commands.add_parser(
        "taxonomy", help="dump the per-layer mixer/FFN taxonomy of one target"
    )
    taxonomy.add_argument("model", choices=sorted(models.REGISTRY))
    _single_device(taxonomy)
    taxonomy.set_defaults(func=cmd_taxonomy)

    rollback = commands.add_parser(
        "rollback", help="check that a rejected block restores target state"
    )
    rollback.add_argument("model", choices=sorted(models.REGISTRY))
    rollback.add_argument("--prompt-tokens", type=int, default=512)
    rollback.add_argument("--block-size", type=int, default=16)
    rollback.add_argument(
        "--accepted", type=int, nargs="+", default=[0, 1, 4, 15],
        help="how many of the block's tokens to accept before rolling back",
    )
    rollback.add_argument("--probe-tokens", type=int, default=4)
    _single_device(rollback)
    rollback.set_defaults(func=cmd_rollback)

    verify = commands.add_parser(
        "verify-width", help="verify cost at q in {1,4,8,16} from one state"
    )
    verify.add_argument("model", choices=sorted(models.REGISTRY))
    verify.add_argument("--input-tokens", type=int, nargs="+", default=[4096, 32768])
    verify.add_argument("--widths", type=int, nargs="+", default=[1, 4, 8, 16])
    verify.add_argument("--repeats", type=int, default=5)
    verify.add_argument(
        "--cache-policies", nargs="+", default=["native", "recording"],
        choices=["native", "recording"],
        help="native is the AR decode path; recording is what DFlash verifies with",
    )
    verify.add_argument("--prompt-task", default=DEFAULT_TASK)
    verify.add_argument("--prompt-seed", type=int, default=42)
    _single_device(verify)
    verify.set_defaults(func=cmd_verifyq)

    run_sweep = commands.add_parser(
        "sweep", help="run the condition list for one model (AR + DFlash)"
    )
    run_sweep.add_argument("model", choices=sorted(models.REGISTRY))
    run_sweep.add_argument(
        "--sweeps", nargs="+", default=["sequence", "block", "output"],
        choices=["sequence", "block", "output"],
    )
    run_sweep.add_argument(
        "--input-tokens", type=int, nargs="*", default=None,
        help="restrict to these input lengths (default: whatever the sweeps imply)",
    )
    run_sweep.add_argument(
        "--passes", nargs="+", default=["perf", "memory"],
        choices=list(DFLASH_PASSES),
        help="trace = torch.profiler device busy/idle and per-module time; "
        "moe = expert routing capture (MoE targets only)",
    )
    run_sweep.add_argument(
        "--ar-policies", nargs="+", default=["native", "recording"],
        choices=["native", "recording"],
    )
    run_sweep.add_argument(
        "--ar-order", default="alternate", choices=["fixed", "alternate"],
        help="alternate reverses the AR policy order on every other condition, "
        "so an order effect shows up as a policy-by-parity difference",
    )
    run_sweep.add_argument("--repeats", type=int, default=3)
    run_sweep.add_argument(
        "--trace-steps", type=int, default=32,
        help="trace pass profiles this many verify (or AR) steps, starting at "
        "the first; prefill and the first draft call are never traced",
    )
    run_sweep.add_argument(
        "--hidden-states", default="selective", choices=["selective", "full"]
    )
    run_sweep.add_argument(
        "--prompt-mode", default=MODE_FIXED_TASK,
        choices=[MODE_FIXED_TASK, MODE_LEGACY],
        help="fixed-task: one task, same source documents at every S "
        "(default); legacy: the old per-S draw across the task mix",
    )
    run_sweep.add_argument("--prompt-task", default=DEFAULT_TASK)
    run_sweep.add_argument(
        "--prompts-per-length", type=int, default=1,
        help="K prompts at every S, so prompt-to-prompt spread is measured",
    )
    run_sweep.add_argument("--prompt-seed", type=int, default=42)
    run_sweep.add_argument("--no-natural", action="store_true")
    _single_device(run_sweep)
    run_sweep.add_argument(
        "--no-fla", action="store_true",
        help="hide fla and causal_conv1d so GDN runs the torch reference "
        "kernels; the record is labelled with the resolved backend",
    )
    run_sweep.set_defaults(func=cmd_sweep)

    breakdown = commands.add_parser(
        "trace-breakdown",
        help="module kind x kernel category device time for one phase of a saved trace",
    )
    breakdown.add_argument("traces", nargs="+")
    breakdown.add_argument("--phase", default="decode: target verify")
    breakdown.add_argument(
        "--out", default=None,
        help="write JSON here (default: <trace>.breakdown.json beside each trace)",
    )
    breakdown.set_defaults(func=cmd_trace_breakdown)

    export = commands.add_parser(
        "export", help="rebuild CSVs from sweep records into csv/<record name>/"
    )
    export.add_argument("records", nargs="+")
    export.add_argument("--out-dir", default=None)
    export.set_defaults(func=cmd_export)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "no_fla", False):
        # Must precede the first import of the Qwen3.5 modeling modules: the
        # kernel fallback is bound when they are imported, not when called.
        sys.modules["fla"] = None
        sys.modules["causal_conv1d"] = None
    if args.command in GPU_COMMANDS:
        if getattr(args, "single_device", False):
            args.num_gpus = 1
        # Before anything initialises CUDA: the visible set is read once.
        args.gpu_selection = env.pin_devices(
            args.num_gpus, uuids=args.gpu_uuids,
            allow_busy=args.allow_busy_gpus,
        )
        print(f"[gpus] {args.num_gpus} pinned: "
              f"{args.gpu_selection['cuda_visible_devices']}", flush=True)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
