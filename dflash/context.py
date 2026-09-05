"""Prompts built to a requested context length, sourced from LongBench.

The benchmark datasets in :mod:`dflash.benchmark` carry whatever prompt length
their source happens to have, which is far too short to say anything about how
DFlash behaves at 1k-16k. This module instead draws from LongBench-E — the
length-balanced split of LongBench — and trims each document so the templated,
tokenized prompt lands on the requested context length.

LongBench v1 still ships as a loading script, which ``datasets`` no longer
executes, so the archive is read directly from the Hub cache. Prompt templates
are the official ones from ``LongBench/config/dataset2prompt.json``.
"""

from __future__ import annotations

import json
import random
import zipfile

LONGBENCH_REPO = "THUDM/LongBench"
LONGBENCH_ARCHIVE = "data.zip"

# Official LongBench prompt templates, restricted to the English LongBench-E
# tasks. ``max_gen`` is LongBench's own generation budget, recorded for
# reference; --max-new-tokens is what actually caps decoding here.
TASKS = {
    "gov_report": {
        "prompt": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "max_gen": 512,
    },
    "multi_news": {
        "prompt": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "max_gen": 512,
    },
    "qasper": {
        "prompt": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "max_gen": 128,
    },
    "multifieldqa_en": {
        "prompt": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 64,
    },
    "lcc": {
        "prompt": "Please complete the code given below. \n{context}Next line of code:\n",
        "max_gen": 64,
    },
    "repobench-p": {
        "prompt": "Please complete the code given below. \n{context}{input}Next line of code:\n",
        "max_gen": 64,
    },
    "hotpotqa": {
        "prompt": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 32,
    },
    "2wikimqa": {
        "prompt": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 32,
    },
    "triviaqa": {
        "prompt": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "max_gen": 32,
    },
    "samsum": {
        "prompt": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "max_gen": 128,
    },
    "trec": {
        "prompt": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "max_gen": 64,
    },
    "passage_count": {
        "prompt": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "max_gen": 32,
    },
    "passage_retrieval_en": {
        "prompt": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        "max_gen": 32,
    },
}

# Summarization, long-form QA and code completion. The remaining LongBench
# tasks answer in a handful of tokens, which leaves too few decode steps for
# per-token latency and acceptance to mean anything.
DEFAULT_TASKS = (
    "gov_report",
    "multi_news",
    "qasper",
    "multifieldqa_en",
    "lcc",
    "repobench-p",
)

# The fit loop below is iterative because decoding a token slice and re-encoding
# it is not token-identical at the boundary. Six rounds is far more than the two
# it takes in practice.
_MAX_FIT_ROUNDS = 6
_LENGTH_TOLERANCE = 16


def resolve_tasks(spec: str | None) -> list[str]:
    """Turn a comma-separated --context-task value into a task list."""
    if spec is None:
        return list(DEFAULT_TASKS)
    if spec == "all":
        return list(TASKS)
    tasks = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in tasks if name not in TASKS]
    if unknown:
        raise ValueError(f"Unknown LongBench task(s) {unknown}. Available: {list(TASKS)}")
    return tasks


def _load_rows(tasks: list[str]) -> list[dict]:
    """Read the LongBench-E jsonl for each task straight out of the archive."""
    from huggingface_hub import hf_hub_download

    archive = zipfile.ZipFile(
        hf_hub_download(LONGBENCH_REPO, LONGBENCH_ARCHIVE, repo_type="dataset")
    )
    rows = []
    for task in tasks:
        with archive.open(f"data/{task}_e.jsonl") as handle:
            for line in handle:
                row = json.loads(line)
                row["task"] = task
                rows.append(row)
    return rows


def _truncate_middle(token_ids: list[int], budget: int) -> list[int]:
    """Keep the head and tail of a document, as LongBench itself does."""
    if len(token_ids) <= budget:
        return token_ids
    head = budget // 2
    return token_ids[:head] + token_ids[len(token_ids) - (budget - head) :]


def _fit_prompt(tokenizer, apply_template, row, context_ids, context_length):
    """Choose a context budget so the full templated prompt hits the target.

    Returns the best (prompt, token_length) pair found, or None when the
    document is too short to reach the target at all.
    """
    template = TASKS[row["task"]]["prompt"]
    budget = context_length
    best = None
    for _ in range(_MAX_FIT_ROUNDS):
        budget = max(1, min(budget, len(context_ids)))
        prompt = apply_template(
            template.format(
                context=tokenizer.decode(_truncate_middle(context_ids, budget)),
                input=row["input"],
            )
        )
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if best is None or abs(length - context_length) < abs(best[1] - context_length):
            best = (prompt, length)
        if length == context_length:
            return best
        shortfall = context_length - length
        if budget + shortfall > len(context_ids):
            # The document cannot cover the target; keep the best near-miss so
            # the caller can decide whether it is close enough.
            break
        budget += shortfall
    return best


def build_dataset(
    tokenizer,
    apply_template,
    context_length: int,
    num_samples: int,
    *,
    tasks: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Build ``num_samples`` LongBench prompts of ``context_length`` tokens each.

    ``apply_template`` renders a user message into the model's chat prompt; the
    fit accounts for the tokens that template adds. Samples are drawn in a
    shuffled order across ``tasks`` so the mix stays balanced.
    """
    rows = _load_rows(tasks or list(DEFAULT_TASKS))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    samples: list[dict] = []
    skipped = 0
    for index in order:
        if len(samples) == num_samples:
            break
        row = rows[index]
        context_ids = tokenizer.encode(row["context"], add_special_tokens=False)
        fitted = _fit_prompt(
            tokenizer, apply_template, row, context_ids, context_length
        )
        if fitted is None or abs(fitted[1] - context_length) > _LENGTH_TOLERANCE:
            skipped += 1
            continue
        prompt, length = fitted
        samples.append(
            {
                "prompt": prompt,
                "num_input_tokens": length,
                "task": row["task"],
                "source_index": index,
            }
        )

    if len(samples) < num_samples:
        raise ValueError(
            f"Only {len(samples)} of {num_samples} LongBench documents reached "
            f"{context_length} tokens ({skipped} skipped). Widen --context-task "
            f"or lower --max-samples."
        )
    if skipped:
        print(f"[context] skipped {skipped} documents shorter than {context_length} tokens")
    return samples
