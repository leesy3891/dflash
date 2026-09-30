"""Fixing the prompt so a length is a controlled variable, not a coincidence.

Every condition in the sweep claims an input length. Two runs that claim the
same length are only comparable if they ran the same tokens, so the prompt is
built once per (model, length), fingerprinted, and the fingerprint goes into
the record. How it was built is recorded too: which LongBench task and split,
whether the context was composed from several documents to reach the length,
and whether it was truncated to fit.

LongBench is the source. Where it cannot be reached -- no network, no cached
dataset -- the sweep does not silently substitute something else and carry on:
it falls back to a declared synthetic filler and marks every record built from
it, because acceptance on repeated filler text is not acceptance on prose and
must never be read as if it were.
"""

from __future__ import annotations

from .env import token_hash, text_hash

SOURCE_LONGBENCH = "longbench"
SOURCE_SYNTHETIC = "synthetic_filler"

# Declared, not incidental. Used only when LongBench is unreachable.
FILLER = (
    "The history of computing is a history of abstractions, each one built to "
    "hide the cost of the layer beneath it and each one eventually measured to "
    "find out what it hid. "
)


def _apply_template(tokenizer, user_content: str, reasoning=None) -> str:
    from ..benchmark import apply_chat_template

    return apply_chat_template(
        tokenizer, [{"role": "user", "content": user_content}], reasoning
    )


MODE_FIXED_TASK = "fixed-task"
MODE_LEGACY = "legacy"
DEFAULT_TASK = "2wikimqa"


def build_set(
    tokenizer,
    lengths,
    *,
    mode: str = MODE_FIXED_TASK,
    task: str = DEFAULT_TASK,
    per_length: int = 1,
    seed: int = 42,
) -> dict[int, list[dict]]:
    """The prompts for a whole S sweep: ``per_length`` of them at every S.

    ``legacy`` is the old behaviour -- one prompt per S drawn from the whole
    LongBench-E task mix -- and exists only to reproduce old records. It is
    what confounded the S axis: 4k landed on multi_news, 16k on trec and the
    rest on 2wikimqa, so a change in acceptance along S was partly a change of
    task.

    ``fixed-task`` holds the task, the split and the extension policy fixed
    across S. With the same seed the shuffled document order is then the same
    at every S, so prompt *i* at every length is built from the same source
    document: middle-truncated where the document is long enough, extended
    with same-task passages where it is not. What still varies with S is the
    amount of context, which is the variable. The record carries each
    prompt's ``source_index`` so that claim can be checked, not trusted.
    """
    out: dict[int, list[dict]] = {}
    for length in lengths:
        if mode == MODE_LEGACY:
            prompts = [build(tokenizer, length, seed=seed)]
        elif mode == MODE_FIXED_TASK:
            prompts = build(
                tokenizer, length, tasks=[task], split="full", extend="on",
                seed=seed, num_samples=per_length,
            )
        else:
            raise ValueError(f"unknown prompt mode {mode!r}")
        if isinstance(prompts, dict):
            prompts = [prompts]
        for index, prompt in enumerate(prompts):
            prompt["prompt_mode"] = mode
            prompt["prompt_index"] = index
        out[length] = prompts
    if mode == MODE_FIXED_TASK:
        # Did every S really get the same source documents?
        by_index: dict[int, set] = {}
        for prompts in out.values():
            for prompt in prompts:
                by_index.setdefault(prompt["prompt_index"], set()).add(
                    (prompt.get("task"), prompt.get("source_index"))
                )
        same = all(len(sources) == 1 for sources in by_index.values())
        for prompts in out.values():
            for prompt in prompts:
                prompt["same_source_across_lengths"] = same
    return out


def build(
    tokenizer,
    context_length: int,
    *,
    tasks: list[str] | None = None,
    split: str = "auto",
    extend: str = "auto",
    seed: int = 42,
    num_samples: int | None = None,
) -> dict | list[dict]:
    """One prompt of ``context_length`` tokens, with its provenance.

    With ``num_samples`` set, a list of that many instead.
    """
    report: dict = {}
    try:
        from .. import context as context_module

        samples = context_module.build_dataset(
            tokenizer,
            lambda content: _apply_template(tokenizer, content),
            context_length,
            num_samples or 1,
            tasks=tasks,
            seed=seed,
            split=split,
            extend=extend,
            report=report,
        )
    except Exception as exc:  # noqa: BLE001 - unreachable dataset is a datum
        samples, failure = [], f"{type(exc).__name__}: {exc}"
    else:
        failure = None if samples else "build_dataset returned no samples"

    if samples and num_samples:
        return [_from_sample(tokenizer, s, context_length, report) for s in samples]
    if samples:
        return _from_sample(tokenizer, samples[0], context_length, report)
    fallback = _fallback(tokenizer, context_length, failure)
    return [fallback] if num_samples else fallback


def _from_sample(tokenizer, sample: dict, context_length: int, report: dict) -> dict:
    prompt = sample["prompt"]
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    return {
        "source": SOURCE_LONGBENCH,
        "prompt": prompt,
        "input_ids": ids,
        "requested_tokens": context_length,
        "actual_tokens": int(ids.shape[1]),
        "task": sample.get("task"),
        "split": sample.get("split"),
        "composed": bool(sample.get("composed")),
        "source_index": sample.get("source_index"),
        "fitted_input_tokens": sample.get("num_input_tokens"),
        "truncated": sample.get("num_input_tokens") != context_length,
        "task_report": report,
        "token_hash": token_hash(ids),
        "prompt_sha": text_hash(prompt),
        "comparable_across_models": False,
        "note": (
            "Tokenisation differs between the Qwen3 and Qwen3.5 vocabularies, "
            "so the same target token count is a different amount of text. "
            "Lengths are comparable within a model, not across models."
        ),
    }


def _fallback(tokenizer, context_length: int, failure) -> dict:
    """Declared loudly: synthetic filler when LongBench cannot be reached."""
    repeats = max(1, context_length // 16)
    text = FILLER * repeats
    ids = tokenizer.encode(
        _apply_template(tokenizer, text), return_tensors="pt", add_special_tokens=False
    )
    truncated = ids.shape[1] > context_length
    ids = ids[:, :context_length]
    return {
    "source": SOURCE_SYNTHETIC,
    "prompt": None,
    "input_ids": ids,
    "requested_tokens": context_length,
    "actual_tokens": int(ids.shape[1]),
    "task": None,
    "split": None,
    "composed": False,
    "truncated": truncated,
    "longbench_error": failure,
    "token_hash": token_hash(ids),
    "comparable_across_models": False,
    "warning": (
        "SYNTHETIC FILLER, not LongBench. Acceptance measured on repeated "
        "text is not acceptance on prose and must not be reported as such. "
        "Latency and memory figures remain valid, since they depend on "
        "token count rather than content."
    ),
    }
