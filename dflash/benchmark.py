from __future__ import annotations

import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from types import SimpleNamespace

import requests
from tqdm import tqdm

MODEL_PRESETS = {
    "qwen3-8b": {
        "model": "Qwen/Qwen3-8B",
        "draft": "z-lab/Qwen3-8B-DFlash-b16",
    },
    "qwen3.5-9b": {
        "model": "Qwen/Qwen3.5-9B",
        "draft": "z-lab/Qwen3.5-9B-DFlash",
    },
}

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


def _reasoning_kwargs(reasoning: str | None, template: str | None = None) -> dict:
    if reasoning is None:
        return {}
    if reasoning in {"on", "off"}:
        if template is not None and "enable_thinking" not in template:
            raise ValueError("This model does not support --reasoning on/off")
        return {"enable_thinking": reasoning == "on"}
    if template is not None:
        for key in ("reasoning_strength", "reasoning_effort"):
            if key in template:
                return {key: reasoning}
        raise ValueError("This model supports only --reasoning on/off")
    return {
        "enable_thinking": True,
        "reasoning_effort": reasoning,
        "reasoning_strength": reasoning,
    }


def apply_chat_template(
    tokenizer,
    messages: list[dict],
    reasoning: str | None,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_reasoning_kwargs(reasoning, str(tokenizer.chat_template or "")),
    )


def _rope_config(config):
    """The sub-config that owns RoPE, unwrapping a text-config container."""
    return getattr(config, "text_config", None) or config


def apply_rope_scaling(
    config, *, rope_type: str = "yarn", factor: float | None = None,
    original_max: int | None = None, needed_positions: int | None = None,
) -> dict:
    """Widen a model's RoPE so positions past its trained window stay in range.

    Qwen3-8B declares 40960 positions and its DFlash draft inherits the same
    number, so a 64k prompt is pure extrapolation for both. YaRN interpolates
    instead, and has to be applied to target and draft identically -- the
    drafter is fed the target's absolute position ids, so the two must agree on
    what a position means. With no explicit ``factor`` the smallest power of two
    that covers ``needed_positions`` is used, never below 2.

    Returns the parameters written, for the record file.
    """
    text = _rope_config(config)
    params = dict(getattr(text, "rope_parameters", None) or {})
    params.setdefault("rope_theta", getattr(text, "rope_theta", 10000.0))
    original = int(original_max or text.max_position_embeddings)
    if factor is None:
        factor = 2.0
        while needed_positions and original * factor < needed_positions:
            factor *= 2
    params.update(
        rope_type=rope_type,
        factor=float(factor),
        original_max_position_embeddings=original,
    )
    text.rope_scaling = params
    text.max_position_embeddings = int(original * float(factor))
    return {**params, "max_position_embeddings": text.max_position_embeddings}


def free_memory_budget(reserve_gb: float = 2.0) -> dict[int, str]:
    """Per-GPU weight budget from what is actually free right now.

    ``device_map="auto"`` plans against each card's *total* memory, which on a
    shared machine hands layers to a GPU someone else is already using. This
    reports free memory instead, less a reserve for the drafter and the forward
    activations that land on top of the weights.
    """
    import torch

    budget = {}
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        budget[index] = f"{max(free_bytes / (1 << 30) - reserve_gb, 0.0):.1f}GiB"
    return budget


def load_transformers_models(
    model_id: str, draft_id: str, device, rope: dict | None = None,
    device_map: str | None = None, max_memory: dict | None = None,
):
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    from .model import DFlash2DraftModel, DFlashDraftModel

    target_kwargs = {"attn_implementation": "sdpa", "dtype": torch.bfloat16}
    target_config = AutoConfig.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(draft_id)
    if rope is not None:
        rope["target"] = apply_rope_scaling(target_config, **rope["request"])
        rope["draft"] = apply_rope_scaling(config, **rope["request"])
        target_kwargs["config"] = target_config

    if device_map is not None:
        # Sharding the target buys headroom for the prefill activation, which
        # at 64k is the term that does not fit. The drafter stays whole on
        # `device`; dflash_generate keeps its bookkeeping there and moves the
        # few tensors that cross a shard boundary explicitly.
        target_kwargs["device_map"] = device_map
        target_kwargs["max_memory"] = (
            max_memory if max_memory is not None else free_memory_budget()
        )

    try:
        target = AutoModelForCausalLM.from_pretrained(model_id, **target_kwargs)
    except ValueError:
        target = AutoModelForImageTextToText.from_pretrained(model_id, **target_kwargs)
    target = (target if device_map is not None else target.to(device)).eval()

    draft_class = (
        DFlash2DraftModel
        if "DFlash2DraftModel" in (config.architectures or [])
        else DFlashDraftModel
    )
    draft = (
        draft_class.from_pretrained(
            draft_id,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
            **({"config": config} if rope is not None else {}),
        )
        .to(device)
        .eval()
    )
    return target, draft, AutoTokenizer.from_pretrained(model_id)


def load_mlx_models(model_id: str, draft_id: str, draft_bits: int | None):
    import mlx.core as mx
    from mlx import nn

    from .model_mlx import load, load_draft

    model, tokenizer = load(model_id)
    draft = load_draft(draft_id)
    if draft_bits is not None:
        nn.quantize(draft, group_size=64, bits=draft_bits)
        mx.eval(draft.parameters())
    return model, draft, tokenizer


def stop_token_ids(model, tokenizer) -> list[int]:
    token_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    return [token_ids] if isinstance(token_ids, int) else list(token_ids)


def send_openai(
    base_url: str,
    messages: list[dict],
    *,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
    reasoning: str | None = None,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": _reasoning_kwargs(reasoning),
        "return_meta_info": True,
    }
    if top_k > 0:
        body["top_k"] = top_k
    response = requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def load_and_process_dataset(data_name: str) -> list[dict]:
    from datasets import load_dataset

    if data_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{data_name}'. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[data_name]
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])
    return [
        {"turns": cfg["format"](row) if cfg.get("multi_turn") else [cfg["format"](row)]}
        for row in dataset
    ]


def _select_dataset(
    dataset: list[dict], count: int | None, *, repeat: bool = False,
) -> list[dict]:
    order = list(range(len(dataset)))
    random.Random(42).shuffle(order)
    count = len(order) if count is None else count
    if not repeat:
        count = min(count, len(order))
    return [dataset[order[i % len(order)]] for i in range(count)]


def _make_decode_metrics(num_output_tokens: int, generation_tps: float, acceptance_lengths: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        num_output_tokens=num_output_tokens,
        time_per_output_token=1.0 / generation_tps if generation_tps > 0 else float("inf"),
        acceptance_lengths=acceptance_lengths,
    )


def _print_decode_summary(responses: list[dict[int, SimpleNamespace]], block_size: int) -> None:
    baseline_tpot = statistics.mean(r[1].time_per_output_token for r in responses)
    dflash_tpot = statistics.mean(r[block_size].time_per_output_token for r in responses)
    print(f"Baseline throughput: {1 / baseline_tpot:.2f} tok/s")
    print(f"DFlash throughput:  {1 / dflash_tpot:.2f} tok/s")
    print(f"Decoding speedup: {baseline_tpot / dflash_tpot:.2f}")

    per_request = [
        r[block_size].acceptance_lengths
        for r in responses
        if r[block_size].acceptance_lengths
    ]
    acceptance_lengths = list(chain.from_iterable(r[block_size].acceptance_lengths for r in responses))
    if not acceptance_lengths:
        print("Average Acceptance length: n/a")
        return
    mean_accept = statistics.mean(statistics.mean(x) for x in per_request)
    print(f"Average Acceptance length: {mean_accept:.2f}")

    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")


def _run_transformers(args: argparse.Namespace) -> None:
    import torch

    from .model import dflash_generate

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    target, draft_model, tokenizer = load_transformers_models(
        args.model, args.draft, device
    )

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    dataset = load_and_process_dataset(args.dataset)

    dataset = _select_dataset(dataset, args.max_samples)

    warmup_text = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": dataset[0]["turns"][0]}],
        args.reasoning,
    )
    warmup = tokenizer.encode(
        warmup_text, return_tensors="pt", add_special_tokens=False
    ).to(device)
    warmup_tokens = min(64, args.max_new_tokens)
    for bs in (1, block_size):
        dflash_generate(
            draft_model, target, warmup, warmup_tokens, None,
            args.temperature, args.top_p, args.top_k, block_size=bs,
        )

    responses = []
    for idx in tqdm(range(len(dataset))):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = apply_chat_template(
                tokenizer, messages, args.reasoning,
            )
            input_ids = tokenizer.encode(
                input_text, return_tensors="pt", add_special_tokens=False
            ).to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    draft_model,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids(target, tokenizer),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    block_size=bs,
                    return_stats=True,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    _print_decode_summary(responses, block_size)


def _run_mlx(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx_lm import stream_generate as stream_generate_baseline

    from .model_mlx import make_sampler, stream_generate

    mx.random.seed(0)
    sampler = make_sampler(args.temperature, args.top_p, args.top_k)

    print(f"Loading target: {args.model}")
    print(f"Loading draft: {args.draft}")
    model, draft, tokenizer = load_mlx_models(
        args.model, args.draft, args.draft_bits
    )
    block_size = args.block_size if args.block_size is not None else int(draft.config.block_size)

    dataset = load_and_process_dataset(args.dataset)
    dataset = _select_dataset(dataset, args.max_samples)

    warmup_prompt = tokenizer.encode("Hi")
    list(stream_generate_baseline(model, tokenizer, warmup_prompt, 3, sampler=sampler))
    list(stream_generate(
        model, draft, tokenizer, warmup_prompt,
        block_size=block_size,
        max_tokens=3,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ))

    responses = []
    for idx in tqdm(range(len(dataset))):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            prompt = apply_chat_template(
                tokenizer, messages, args.reasoning,
            )

            response = {}

            tokens_bl, tps_bl = [], 0
            for r in stream_generate_baseline(model, tokenizer, prompt, args.max_new_tokens, sampler=sampler):
                tokens_bl.append(r.token)
                tps_bl = r.generation_tps
            response[1] = _make_decode_metrics(len(tokens_bl), tps_bl, [1])

            tokens_df, accs, tps_df = [], [], 0
            for r in stream_generate(
                model, draft, tokenizer, prompt,
                block_size=block_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ):
                tokens_df.extend(r.tokens)
                if r.accepted is not None:
                    accs.append(r.accepted)
                tps_df = r.generation_tps
            response[block_size] = _make_decode_metrics(len(tokens_df), tps_df, accs)

            output_text = tokenizer.decode(tokens_df)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    _print_decode_summary(responses, block_size)


def _run_openai(args: argparse.Namespace) -> None:
    bs = max(args.concurrency, 1)
    dataset = _select_dataset(
        load_and_process_dataset(args.dataset), args.num_prompts + bs, repeat=True,
    )
    prompts = [
        [{"role": "user", "content": item["turns"][0]}]
        for item in dataset[:args.num_prompts]
    ]
    warmup_prompts = [
        [{"role": "user", "content": item["turns"][0]}]
        for item in dataset[args.num_prompts:]
    ]

    def send_one(messages: list[dict], max_new_tokens=args.max_new_tokens) -> dict:
        return send_openai(
            args.base_url,
            messages,
            model=args.model,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout_s=args.timeout_s,
            reasoning=args.reasoning,
        )

    print(f"[warmup] {bs} requests ...")
    with ThreadPoolExecutor(max_workers=bs) as pool:
        list(pool.map(lambda p: send_one(p, min(64, args.max_new_tokens)), warmup_prompts))

    print(f"Running benchmark: {args.num_prompts} prompts, concurrency={bs} ...")
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    with ThreadPoolExecutor(max_workers=bs) as pool:
        futures = [pool.submit(send_one, p) for p in prompts]
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="Benchmarking"):
            out = fut.result()
            usage = out.get("usage", {}) or {}
            total_tokens += int(usage.get("completion_tokens", 0))
            meta = out.get("meta_info", {}) or {}
            spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            if "spec_accept_length" in meta:
                try:
                    spec_accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    print(f"\n{'=' * 50}")
    print(f"Backend:          {args.backend}")
    print(f"Dataset:          {args.dataset}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Concurrency:      {bs}")
    print(f"Latency:          {latency:.1f}s")
    print(f"Output tokens:    {total_tokens}")
    print(f"Throughput:       {toks_per_s:,.2f} tok/s")
    if spec_accept_lengths:
        print(f"Accept length:    {statistics.mean(spec_accept_lengths):.3f}")
    if spec_verify_ct_sum > 0:
        print(f"Spec verify ct:   {spec_verify_ct_sum}")
    print(f"{'=' * 50}")


def _build_context_samples(args, tokenizer) -> tuple[list[dict], list[str], dict]:
    """Fit the LongBench prompts for a context-length run and report the mix."""
    from . import context as context_module

    def apply_template(user_content: str) -> str:
        return apply_chat_template(
            tokenizer, [{"role": "user", "content": user_content}], args.reasoning
        )

    num_samples = args.max_samples if args.max_samples is not None else 32
    tasks = context_module.resolve_tasks(args.context_task, args.context_length)
    report: dict = {}
    samples = context_module.build_dataset(
        tokenizer,
        apply_template,
        args.context_length,
        num_samples,
        tasks=tasks,
        split=args.context_split,
        extend=args.context_extend,
        report=report,
    )
    context_module.print_report(report, args.context_length)
    return samples, tasks, report


def _parse_max_memory(spec: str | None) -> dict | None:
    """Parse ``--max-memory "0=20GiB,1=32GiB"`` into accelerate's mapping."""
    if not spec:
        return None
    budget = {}
    for item in spec.split(","):
        index, _, size = item.partition("=")
        budget[int(index.strip())] = size.strip()
    return budget


def _max_position_embeddings(config) -> int | None:
    """The target's trained position budget, through a text-config wrapper."""
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is None:
            continue
        value = getattr(candidate, "max_position_embeddings", None)
        if value:
            return int(value)
    return None


def _run_context_dry_run(args: argparse.Namespace) -> None:
    """Fit the prompts and print the feasibility table without loading models."""
    from transformers import AutoTokenizer

    from . import context as context_module

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples, _, _ = _build_context_samples(args, tokenizer)
    lengths = [sample["num_input_tokens"] for sample in samples]
    composed = sum(sample["composed"] for sample in samples)
    print(
        f"\n{len(samples)} prompts, {min(lengths)}-{max(lengths)} tokens, "
        f"{len(samples) - composed} natural / {composed} composed"
    )
    print(f"extend={context_module.resolve_extend(args.context_extend, args.context_length)}")


def _run_context_length(args: argparse.Namespace) -> None:
    """Benchmark at a fixed input context length and write a record file."""
    import torch

    from . import context as context_module
    from . import record as record_module
    from .model import dflash_generate, module_bytes

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    rope = (
        {
            "request": {
                "rope_type": args.rope_scaling,
                "factor": args.rope_factor,
                "original_max": args.rope_original_max,
                "needed_positions": args.context_length + args.max_new_tokens,
            }
        }
        if args.rope_scaling != "none"
        else None
    )
    target, draft_model, tokenizer = load_transformers_models(
        args.model, args.draft, device, rope=rope, device_map=args.device_map,
        max_memory=_parse_max_memory(args.max_memory),
    )
    if args.device_map is not None:
        print(f"[device-map] target sharded: {getattr(target, 'hf_device_map', {})}")
    if rope is not None:
        print(f"[rope] target {rope['target']}")
        print(f"[rope] draft  {rope['draft']}")
    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    samples, tasks, task_report = _build_context_samples(args, tokenizer)

    # RoPE extrapolates past the trained window without complaining, so say so
    # rather than let a silently-out-of-range run look like a measurement.
    position_limit = _max_position_embeddings(target.config)
    beyond_position_limit = (
        position_limit is not None
        and args.context_length + args.max_new_tokens > position_limit
    )
    if beyond_position_limit:
        print(
            f"[warning] {args.model} was trained to {position_limit} positions; "
            f"{args.context_length} + {args.max_new_tokens} goes past it. Both "
            f"target and draft are extrapolating, so acceptance at this length "
            f"is not comparable with the rest of the sweep. Consider "
            f"--rope-scaling yarn."
        )

    def encode(prompt: str):
        return tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(device)

    configs = {"dflash": block_size}
    if args.baseline:
        configs["baseline"] = 1

    # Warm up at the real context length so allocator growth and kernel
    # autotuning do not land inside the measured samples.
    warmup_ids = encode(samples[0]["prompt"])
    for size in configs.values():
        dflash_generate(
            draft_model, target, warmup_ids, min(64, args.max_new_tokens), None,
            args.temperature, args.top_p, args.top_k, block_size=size,
            hidden_states=args.hidden_states,
            prefill_chunk=args.prefill_chunk,
        )

    stop = stop_token_ids(target, tokenizer)
    runs: dict[str, list[dict]] = {name: [] for name in configs}
    per_sample = []
    for index, sample in enumerate(tqdm(samples, desc=f"ctx={args.context_length}")):
        input_ids = encode(sample["prompt"])
        entry = {
            "index": index,
            "task": sample["task"],
            "split": sample["split"],
            "composed": sample["composed"],
            "source_index": sample["source_index"],
            "fitted_input_tokens": sample["num_input_tokens"],
        }
        for name, size in configs.items():
            torch.cuda.reset_peak_memory_stats()
            stats = dflash_generate(
                draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                block_size=size,
                return_stats=True,
                profile_draft_memory=args.profile_draft_memory and name == "dflash",
                hidden_states=args.hidden_states,
                prefill_chunk=args.prefill_chunk,
            )
            metrics = record_module.sample_metrics(stats, drafter=name == "dflash")
            runs[name].append(metrics)
            entry[name] = metrics
        per_sample.append(entry)

    draft_weight_bytes = module_bytes(draft_model)
    summaries = {
        name: record_module.summarize(
            values, block_size, draft_weight_bytes, drafter=name == "dflash"
        )
        for name, values in runs.items()
    }
    record_module.print_summary(summaries, block_size)

    payload = {
        "backend": "transformers",
        "model": args.model,
        "draft": args.draft,
        "model_name": args.model_name,
        "context_length": args.context_length,
        "context_source": context_module.LONGBENCH_REPO,
        "context_tasks": tasks,
        "context_split": args.context_split,
        "context_extend": context_module.resolve_extend(
            args.context_extend, args.context_length
        ),
        "context_task_report": task_report,
        "num_composed_samples": sum(s["composed"] for s in samples),
        "hidden_states": args.hidden_states,
        "prefill_chunk": args.prefill_chunk,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "target_device_map": getattr(target, "hf_device_map", None),
        "num_devices": torch.cuda.device_count(),
        "max_position_embeddings": position_limit,
        "beyond_position_limit": beyond_position_limit,
        "rope_scaling": None if rope is None else {
            "target": rope["target"], "draft": rope["draft"]
        },
        "block_size": block_size,
        "gamma": block_size - 1,
        "num_samples": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "reasoning": args.reasoning,
        "baseline": args.baseline,
        "profile_draft_memory": args.profile_draft_memory,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "summary": summaries,
        "samples": per_sample,
    }
    if args.baseline:
        payload["decoding_speedup"] = (
            summaries["baseline"]["aggregate_time_per_output_token_s"]
            / summaries["dflash"]["aggregate_time_per_output_token_s"]
        )
    path = record_module.write(
        args.record_dir, args.model_name, args.context_length, payload
    )
    print(f"Record written to {path}")


def run(args: argparse.Namespace) -> None:
    if getattr(args, "context_length", None) is not None:
        if getattr(args, "context_dry_run", False):
            _run_context_dry_run(args)
        else:
            _run_context_length(args)
    elif args.backend == "transformers":
        _run_transformers(args)
    elif args.backend == "mlx":
        _run_mlx(args)
    else:
        _run_openai(args)
