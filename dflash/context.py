"""Prompts built to a requested context length, sourced from LongBench.

The benchmark datasets in :mod:`dflash.benchmark` carry whatever prompt length
their source happens to have, which is far too short to say anything about how
DFlash behaves at 1k-64k. This module instead draws from LongBench and trims —
or, past the point where LongBench documents run out, composes — each document
so the templated, tokenized prompt lands on the requested context length.

Two regimes, split at :data:`NATURAL_MAX_CONTEXT`:

* **<= 16k** uses LongBench-E, the length-balanced split, exactly as published:
  one document per prompt, middle-truncated to the target. This is what the
  4k/8k/16k records were produced with and it is unchanged.
* **> 16k** uses the full LongBench split, which is where the long documents
  live, and for tasks whose context is a sequence of independent units it
  appends units drawn from other documents of the same task until the target is
  reached. LongBench caps its own contexts well below 64k -- see the
  feasibility table in PROFILING.md -- so without this no task reaches 64k at
  all, and only NarrativeQA reaches 32k in quantity.

LongBench v1 still ships as a loading script, which ``datasets`` no longer
executes, so the archive is read directly from the Hub cache. Prompt templates
are the official ones from ``LongBench/config/dataset2prompt.json``.
"""

from __future__ import annotations

import json
import random
import re
import zipfile

LONGBENCH_REPO = "THUDM/LongBench"
LONGBENCH_ARCHIVE = "data.zip"

# LongBench-E's longest bucket is "8k+" and in practice nothing in it clears
# ~41k tokens; the full split's longest English document is a 65301-token
# NarrativeQA story. This is the boundary between "a real document reached this
# length" and "units had to be appended to reach it".
NATURAL_MAX_CONTEXT = 16384

# How a task's context decomposes into independently shufflable units, for the
# tasks where that is meaningful. ``marker`` matches the start of each unit and
# is kept with it, so joining the units back with "" reproduces the original
# context byte for byte. ``renumber`` rewrites that marker once the units have
# been reordered; tasks whose units are not numbered leave it None.
_PASSAGE_UNIT = {"marker": r"(?m)^Passage \d+:\n", "renumber": "Passage {n}:\n"}
_PARAGRAPH_UNIT = {"marker": r"(?m)^Paragraph \d+: ", "renumber": "Paragraph {n}: "}

# Official LongBench prompt templates for the English tasks. ``max_gen`` is
# LongBench's own generation budget, recorded for reference; --max-new-tokens is
# what actually caps decoding here. ``splits`` lists the archive members that
# exist for the task -- "e" is the length-balanced LongBench-E file, "full" the
# original one. ``unit`` marks the tasks that can be composed past their
# natural length.
TASKS = {
    "narrativeqa": {
        "prompt": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "max_gen": 128,
        "splits": ("full",),
        "unit": None,
    },
    "gov_report": {
        "prompt": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "max_gen": 512,
        "splits": ("e", "full"),
        "unit": None,
    },
    "qmsum": {
        "prompt": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "max_gen": 512,
        "splits": ("full",),
        "unit": None,
    },
    "multi_news": {
        "prompt": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "max_gen": 512,
        "splits": ("e", "full"),
        "unit": _PASSAGE_UNIT,
    },
    "qasper": {
        "prompt": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "max_gen": 128,
        "splits": ("e", "full"),
        "unit": None,
    },
    "multifieldqa_en": {
        "prompt": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 64,
        "splits": ("e", "full"),
        "unit": None,
    },
    "lcc": {
        "prompt": "Please complete the code given below. \n{context}Next line of code:\n",
        "max_gen": 64,
        "splits": ("e", "full"),
        "unit": None,
    },
    "repobench-p": {
        "prompt": "Please complete the code given below. \n{context}{input}Next line of code:\n",
        "max_gen": 64,
        "splits": ("e", "full"),
        "unit": None,
    },
    "hotpotqa": {
        "prompt": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 32,
        "splits": ("e", "full"),
        "unit": _PASSAGE_UNIT,
    },
    "2wikimqa": {
        "prompt": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 32,
        "splits": ("e", "full"),
        "unit": _PASSAGE_UNIT,
    },
    "musique": {
        "prompt": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 32,
        "splits": ("full",),
        "unit": _PASSAGE_UNIT,
    },
    "triviaqa": {
        "prompt": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "max_gen": 32,
        "splits": ("e", "full"),
        "unit": {"marker": r"(?m)^Passage:\n", "renumber": None},
    },
    "samsum": {
        "prompt": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "max_gen": 128,
        "splits": ("e", "full"),
        "unit": {"marker": r"(?m)^Dialogue: ", "renumber": None},
    },
    "trec": {
        "prompt": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "max_gen": 64,
        "splits": ("e", "full"),
        "unit": {"marker": r"(?m)^Question: ", "renumber": None},
    },
    "passage_count": {
        "prompt": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "max_gen": 32,
        "splits": ("e", "full"),
        "unit": _PARAGRAPH_UNIT,
    },
    "passage_retrieval_en": {
        "prompt": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        # LongBench hardcodes "30 paragraphs" because every one of its own
        # documents has exactly 30. Composing to 32k/64k changes that, so a
        # composed prompt states the count of the context actually built. The
        # official wording above is what the natural lengths still use.
        "prompt_composed": "Here are {num_paragraphs} paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        "max_gen": 32,
        "splits": ("e", "full"),
        "unit": _PARAGRAPH_UNIT,
    },
}

# The 13 English tasks LongBench-E ships. This is what --context-task defaults
# to at or below NATURAL_MAX_CONTEXT, and it is the set the 4k/8k/16k records
# were built from.
LONGBENCH_E_TASKS = tuple(
    name for name, spec in TASKS.items() if "e" in spec["splits"]
)

# Every English task, LongBench-E or not.
ENGLISH_TASKS = tuple(TASKS)

# The three tasks the DFlash paper reports long-context acceptance length on
# (Table 4). Selecting one of these reproduces a single column of that table.
PAPER_TASKS = ("hotpotqa", "qasper", "gov_report")

# The default above NATURAL_MAX_CONTEXT: the tasks that either carry documents
# long enough on their own (narrativeqa, gov_report, qmsum) or decompose into
# units that can be appended to reach the target. Qasper, MultiFieldQA-en, LCC
# and RepoBench-P are left out -- they are single continuous documents that top
# out well short of 32k, so at these lengths they would contribute nothing.
LONG_CONTEXT_TASKS = (
    "narrativeqa",
    "gov_report",
    "qmsum",
    "musique",
    "hotpotqa",
    "2wikimqa",
    "passage_retrieval_en",
    "passage_count",
    "triviaqa",
    "trec",
    "samsum",
)

TASK_GROUPS = {
    "all-e": LONGBENCH_E_TASKS,
    "all-en": ENGLISH_TASKS,
    "paper": PAPER_TASKS,
    "long": LONG_CONTEXT_TASKS,
}

# With no --context-task the whole English LongBench-E suite is used.
DEFAULT_TASKS = LONGBENCH_E_TASKS

# The fit loop below is iterative because decoding a token slice and re-encoding
# it is not token-identical at the boundary. Six rounds is far more than the two
# it takes in practice.
_MAX_FIT_ROUNDS = 6
_LENGTH_TOLERANCE = 16


def composable_tasks() -> tuple[str, ...]:
    """Tasks whose context is a sequence of units that can be extended."""
    return tuple(name for name, spec in TASKS.items() if spec["unit"] is not None)


def resolve_tasks(spec: str | None, context_length: int | None = None) -> list[str]:
    """Turn a --context-task value into a task list.

    ``None`` (the default) picks the group that suits the requested length: the
    English LongBench-E suite at or below :data:`NATURAL_MAX_CONTEXT`, and
    :data:`LONG_CONTEXT_TASKS` above it, which drops the tasks that cannot reach
    32k either naturally or by composition. ``all`` is a spelling of the same
    thing; ``all-e``, ``all-en``, ``paper`` and ``long`` name the groups
    explicitly. Otherwise the value is one task name, or a comma-separated list
    of them -- pass one of ``PAPER_TASKS`` to line up with a column of the
    paper's long-context table.
    """
    if spec is None or spec == "all":
        long_run = context_length is not None and context_length > NATURAL_MAX_CONTEXT
        return list(LONG_CONTEXT_TASKS if long_run else LONGBENCH_E_TASKS)
    if spec in TASK_GROUPS:
        return list(TASK_GROUPS[spec])
    tasks = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in tasks if name not in TASKS]
    if unknown:
        raise ValueError(
            f"Unknown LongBench task(s) {unknown}. Available: {list(TASKS)}; "
            f"groups: {sorted(TASK_GROUPS)}"
        )
    return tasks


def resolve_split(task: str, context_length: int, preference: str = "auto") -> str:
    """Pick the archive member to read a task from.

    ``auto`` keeps LongBench-E for the lengths it covers, so runs at or below
    :data:`NATURAL_MAX_CONTEXT` are unchanged, and switches to the full split
    above it, where the long documents are.
    """
    available = TASKS[task]["splits"]
    if preference == "e":
        if "e" not in available:
            raise ValueError(f"LongBench-E has no split for '{task}'")
        return "e"
    if preference == "full":
        return "full"
    if preference != "auto":
        raise ValueError(f"Unknown --context-split '{preference}'")
    if "e" in available and context_length <= NATURAL_MAX_CONTEXT:
        return "e"
    return "full"


def resolve_extend(preference: str, context_length: int) -> bool:
    """Decide whether composable tasks may be extended past their length.

    ``auto`` turns composition on only above :data:`NATURAL_MAX_CONTEXT`, which
    is the point where LongBench stops supplying documents that long.
    """
    if preference == "on":
        return True
    if preference == "off":
        return False
    if preference != "auto":
        raise ValueError(f"Unknown --context-extend '{preference}'")
    return context_length > NATURAL_MAX_CONTEXT


def _load_rows(tasks: list[str], context_length: int, split: str) -> list[dict]:
    """Read the LongBench jsonl for each task straight out of the archive."""
    from huggingface_hub import hf_hub_download

    archive = zipfile.ZipFile(
        hf_hub_download(LONGBENCH_REPO, LONGBENCH_ARCHIVE, repo_type="dataset")
    )
    rows = []
    for task in tasks:
        chosen = resolve_split(task, context_length, split)
        member = f"data/{task}{'_e' if chosen == 'e' else ''}.jsonl"
        with archive.open(member) as handle:
            for line in handle:
                row = json.loads(line)
                row["task"] = task
                row["split"] = chosen
                rows.append(row)
    return rows


def _truncate_middle(token_ids: list[int], budget: int) -> list[int]:
    """Keep the head and tail of a document, as LongBench itself does."""
    if len(token_ids) <= budget:
        return token_ids
    head = budget // 2
    return token_ids[:head] + token_ids[len(token_ids) - (budget - head) :]


def split_units(task: str, context: str) -> list[str]:
    """Cut a context into its independently shufflable units.

    Each unit keeps its own leading marker and trailing whitespace, so
    ``"".join(split_units(task, c)) == c``. Text before the first marker, which
    the LongBench contexts do not have but a re-encoded one might, is folded
    into the first unit.
    """
    unit = TASKS[task]["unit"]
    if unit is None:
        return [context]
    starts = [match.start() for match in re.finditer(unit["marker"], context)]
    if not starts:
        return [context]
    starts[0] = 0
    bounds = starts + [len(context)]
    return [context[bounds[i] : bounds[i + 1]] for i in range(len(starts))]


def _renumber(task: str, units: list[str]) -> list[str]:
    """Rewrite ``Passage N:`` / ``Paragraph N:`` markers to run 1..len(units)."""
    unit = TASKS[task]["unit"]
    if unit is None or unit["renumber"] is None:
        return units
    marker = re.compile(unit["marker"])
    return [
        marker.sub(unit["renumber"].format(n=index + 1), text, count=1)
        for index, text in enumerate(units)
    ]


class _UnitPool:
    """Donor units for one task, cached with their token lengths.

    Composition draws from every other document of the same task, so a 64k
    prompt for a task whose documents are 15k is roughly four documents' worth
    of units with one document's evidence scattered through it.
    """

    def __init__(self, tokenizer, task: str, rows: list[dict]):
        self.tokenizer = tokenizer
        self.task = task
        self.units: list[str] = []
        self.owner: list[int] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            for text in split_units(task, row["context"]):
                if text in seen:
                    continue
                seen.add(text)
                self.units.append(text)
                self.owner.append(index)
        self._lengths: dict[int, int] = {}

    def length(self, index: int) -> int:
        cached = self._lengths.get(index)
        if cached is None:
            cached = len(
                self.tokenizer.encode(self.units[index], add_special_tokens=False)
            )
            self._lengths[index] = cached
        return cached


def _compose_context(
    tokenizer, task: str, row_index: int, own: list[str], pool: _UnitPool,
    budget: int, rng: random.Random,
) -> str:
    """Build a context of about ``budget`` tokens by appending donor units.

    The document's own units keep their relative order but are scattered
    through the result rather than sitting in a block at the head, so evidence
    is not systematically in the first few thousand tokens. Donors come from
    other documents of the same task; the tail is trimmed to land on budget, so
    the last unit may be cut mid-way exactly as middle-truncation cuts one.
    """
    own_tokens = sum(
        len(tokenizer.encode(text, add_special_tokens=False)) for text in own
    )
    candidates = [i for i, owner in enumerate(pool.owner) if owner != row_index]
    rng.shuffle(candidates)

    donors: list[str] = []
    total = own_tokens
    for index in candidates:
        if total >= budget:
            break
        donors.append(pool.units[index])
        total += pool.length(index)
    if not donors:
        return tokenizer.decode(
            _truncate_middle(
                tokenizer.encode("".join(own), add_special_tokens=False), budget
            )
        )

    # Scatter the document's own units among the donors, never in the final
    # slot -- that one gets trimmed to hit the budget.
    slots = sorted(rng.sample(range(len(donors) + len(own) - 1), len(own)))
    sequence: list[str] = []
    own_iter = iter(own)
    donor_iter = iter(donors)
    for position in range(len(donors) + len(own)):
        sequence.append(next(own_iter) if position in slots else next(donor_iter))

    composed = "".join(_renumber(task, sequence))
    ids = tokenizer.encode(composed, add_special_tokens=False)
    return tokenizer.decode(ids[:budget]) if len(ids) > budget else composed


def _fit_prompt(
    tokenizer, apply_template, row, context_ids, context_length,
    *, render=None,
):
    """Choose a context budget so the full templated prompt hits the target.

    Returns the best (prompt, token_length) pair found, or None when the
    document is too short to reach the target at all. ``render`` builds the
    context at a given token budget: middle-truncation of the document by
    default, unit composition for an extended task.
    """
    spec = TASKS[row["task"]]
    if render is None:
        template = spec["prompt"]

        def render(budget):
            return tokenizer.decode(_truncate_middle(context_ids, budget))
        ceiling = len(context_ids)
    else:
        template = spec.get("prompt_composed", spec["prompt"])
        ceiling = None  # composition can always supply more units

    budget = context_length
    best = None
    for _ in range(_MAX_FIT_ROUNDS):
        budget = max(1, budget if ceiling is None else min(budget, ceiling))
        context = render(budget)
        fields = {"context": context, "input": row["input"]}
        if "{num_paragraphs}" in template:
            fields["num_paragraphs"] = len(split_units(row["task"], context))
        prompt = apply_template(template.format(**fields))
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if best is None or abs(length - context_length) < abs(best[1] - context_length):
            best = (prompt, length)
        if length == context_length:
            return best
        shortfall = context_length - length
        if ceiling is not None and budget + shortfall > ceiling:
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
    split: str = "auto",
    extend: str | bool = "auto",
    report: dict | None = None,
) -> list[dict]:
    """Build ``num_samples`` LongBench prompts of ``context_length`` tokens each.

    ``apply_template`` renders a user message into the model's chat prompt; the
    fit accounts for the tokens that template adds. Samples are drawn in a
    shuffled order across ``tasks`` so the mix stays balanced. Documents that
    cannot reach ``context_length`` are skipped, unless the task is composable
    and extension is enabled, in which case units from other documents of the
    same task make up the difference. ``report``, if given, is filled with a
    per-task breakdown of how many prompts each task produced and how.
    """
    task_list = list(tasks or DEFAULT_TASKS)
    extending = (
        extend if isinstance(extend, bool) else resolve_extend(extend, context_length)
    )
    rows = _load_rows(task_list, context_length, split)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    by_task: dict[str, list[int]] = {task: [] for task in task_list}
    for index, row in enumerate(rows):
        by_task[row["task"]].append(index)
    pools: dict[str, _UnitPool] = {}
    # Row index -> its position within its own task, which is how _UnitPool
    # labels the owner of each unit.
    local_index = {
        index: position
        for task in task_list
        for position, index in enumerate(by_task[task])
    }

    stats = {
        task: {
            "split": resolve_split(task, context_length, split),
            "documents": len(by_task[task]),
            "composable": TASKS[task]["unit"] is not None,
            "natural": 0,
            "composed": 0,
            "skipped": 0,
        }
        for task in task_list
    }

    samples: list[dict] = []
    for index in order:
        if len(samples) == num_samples:
            break
        row = rows[index]
        task = row["task"]
        context_ids = tokenizer.encode(row["context"], add_special_tokens=False)
        composed = (
            extending
            and TASKS[task]["unit"] is not None
            and len(context_ids) < context_length
        )
        render = None
        if composed:
            pool = pools.get(task)
            if pool is None:
                pool = _UnitPool(tokenizer, task, [rows[i] for i in by_task[task]])
                pools[task] = pool
            own = split_units(task, row["context"])
            owner = local_index[index]
            # A fresh Random from the same seed on every fit round, so the
            # composition is a pure function of the budget and the outer loop
            # converges instead of chasing a context that keeps changing.
            sample_seed = seed * 1_000_003 + index

            def render(
                budget, pool=pool, own=own, owner=owner, task=task,
                sample_seed=sample_seed,
            ):
                return _compose_context(
                    tokenizer, task, owner, own, pool, budget,
                    random.Random(sample_seed),
                )

        fitted = _fit_prompt(
            tokenizer, apply_template, row, context_ids, context_length, render=render
        )
        if fitted is None or abs(fitted[1] - context_length) > _LENGTH_TOLERANCE:
            stats[task]["skipped"] += 1
            continue
        prompt, length = fitted
        stats[task]["composed" if composed else "natural"] += 1
        samples.append(
            {
                "prompt": prompt,
                "num_input_tokens": length,
                "task": task,
                "split": row["split"],
                "composed": composed,
                "source_index": index,
            }
        )

    if report is not None:
        report.clear()
        report.update(stats)

    skipped = sum(entry["skipped"] for entry in stats.values())
    if len(samples) < num_samples:
        raise ValueError(
            f"Only {len(samples)} of {num_samples} LongBench documents reached "
            f"{context_length} tokens ({skipped} skipped). Per task: "
            f"{format_report(stats)}. Widen --context-task, enable "
            f"--context-extend on, or lower --max-samples."
        )
    if skipped:
        print(f"[context] skipped {skipped} documents shorter than {context_length} tokens")
    return samples


def format_report(stats: dict) -> str:
    """One-line per-task summary of what build_dataset managed to produce."""
    parts = []
    for task, entry in stats.items():
        if not (entry["natural"] or entry["composed"]):
            continue
        detail = f"{entry['natural']}n"
        if entry["composed"]:
            detail += f"+{entry['composed']}c"
        parts.append(f"{task}={detail}")
    return ", ".join(parts) if parts else "(nothing)"


def print_report(stats: dict, context_length: int) -> None:
    """Feasibility table: what each task contributed at this context length."""
    print(f"\nLongBench feasibility at {context_length} tokens")
    print(f"{'task':22s} {'split':6s} {'docs':>6s} {'natural':>8s} {'composed':>9s} {'skipped':>8s}")
    for task, entry in stats.items():
        print(
            f"{task:22s} {entry['split']:6s} {entry['documents']:6d} "
            f"{entry['natural']:8d} {entry['composed']:9d} {entry['skipped']:8d}"
        )
