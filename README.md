# DFlash: Block Diffusion for Flash Speculative Decoding

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.

<details open>
<summary><strong>DFlash 2</strong></summary>

[**Blog**](https://inco.ai/blog/dflash2/) | [**Models**](https://huggingface.co/collections/z-lab/dflash-2)

<p align="center"><img src="https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash2_system.png" alt="DFlash 2 architecture"></p>

https://github.com/user-attachments/assets/f786e7c5-c2bc-47d4-8a32-1f730a689e1b
</details>

<details>
<summary><strong>DFlash</strong></summary>

[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

![DFlash architecture](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c
</details>

## Supported Models

### DFlash 2

Available checkpoints: [Muse-Glimmer-30B](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2) and [Qwen3.8-27B](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). See the [DFlash 2 collection](https://huggingface.co/collections/z-lab/dflash-2) for updates.

### DFlash

Public checkpoints are available in the [DFlash collection](https://huggingface.co/collections/z-lab/dflash):

- **Qwen:** Qwen3.6 (27B, 35B-A3B), Qwen3.5 (4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B), Qwen3 (4B/8B non-thinking, Coder-Next, Coder-30B-A3B)
- **Gemma:** Gemma 4 (12B, 31B, 26B-A4B)
- **MiniMax:** M2.5, M2.7
- **Kimi:** K2.5, K2.6, K2.7-Code
- **Others:** GPT-OSS (20B, 120B), Llama-3.1-8B, GLM 5.1, Alpamayo 1.5/R1 10B

Use the Transformers or MLX backends below for their explicitly listed model families. Other checkpoints can be benchmarked through an OpenAI-compatible SGLang or vLLM server.

## 📦 Installation

Install the base package for an OpenAI-compatible server, or include local
inference dependencies. The local install uses MLX on Apple Silicon and
Transformers on Linux.

```bash
pip install dflash
pip install "dflash[local]"  # local inference
```

For serving benchmarks, install a supported version of [SGLang](https://github.com/sgl-project/sglang/pull/35371), [vLLM](https://github.com/vllm-project/vllm/pull/52816), [oMLX](https://github.com/z-lab/omlx-fork/releases/download/0.6.2-dflash2/oMLX-0.6.2-zlab-dflash2-arm64-signed.dmg), or [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/27342) separately, launch its OpenAI-compatible server with DFlash, and pass its `--base-url` below.

## 🚀 Quick Start

### Transformers

The Transformers backend supports DFlash 2 for Muse-Glimmer-30B, and DFlash for
Qwen3 and LLaMA-3.1-8B. Muse uses `reasoning_strength`: `low`, `medium`, `high`
(default), or `xhigh`.

```bash
dflash generate transformers \
    --model meta-models/Muse-Glimmer-30B \
    --draft z-lab/Muse-Glimmer-30B-DFlash2 \
    --reasoning high --temperature 1 --top-p 0.95 --top-k 64 \
    "How many positive whole-number divisors does 196 have?"
```

### MLX (Apple Silicon)

The MLX backend supports DFlash 2 for Qwen3.8-27B, and DFlash for Qwen3,
Qwen3.5, Qwen3.6, and Gemma 4. Qwen3.8 uses `reasoning_effort`: `low`, `medium`,
or `xhigh` (default). For quantized targets or drafts, use `block_size <= 5`: MLX's current
quantized matmul kernel becomes less efficient at larger verify widths.
The example below runs both the target and draft with 4-bit weights.

```bash
dflash generate mlx \
    --model mlx-community/Qwen3.8-27B-4bit \
    --draft z-lab/Qwen3.8-27B-DFlash2 \
    --draft-bits 4 --block-size 5 --reasoning xhigh \
    "How many positive whole-number divisors does 196 have?"
```

### OpenAI-compatible server

Launch the latest SGLang or vLLM server separately, then run:

```bash
dflash generate openai \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B \
    "How many positive whole-number divisors does 196 have?"
```

## 📊 Evaluation

All benchmarks share the same datasets (gsm8k, math500, humaneval, mbpp, mt-bench), downloaded and cached by Hugging Face Datasets.

**OpenAI-compatible server** (SGLang or vLLM):
```bash
dflash benchmark openai \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --reasoning xhigh \
    --temperature 1 --top-p 0.95 --top-k 20
```

**Transformers** (Muse-Glimmer-30B DFlash 2):
```bash
dflash benchmark transformers \
    --model meta-models/Muse-Glimmer-30B --draft z-lab/Muse-Glimmer-30B-DFlash2 \
    --dataset gsm8k --max-samples 128 --reasoning high
```

**MLX** (Qwen3.8-27B 4-bit DFlash 2):
```bash
dflash benchmark mlx \
    --model mlx-community/Qwen3.8-27B-4bit --draft z-lab/Qwen3.8-27B-DFlash2 \
    --dataset gsm8k --max-samples 128 --reasoning xhigh --block-size 5 --draft-bits 4
```

### Context-length sweep

Measures DFlash against a `block_size=1` baseline at a fixed input length.
Prompts come from [LongBench-E](https://huggingface.co/datasets/THUDM/LongBench),
the length-balanced split of LongBench, using its official per-task prompt
templates; each document is middle-truncated so the templated prompt lands
exactly on `--context-length`. Transformers backend only. `--model-preset`
fills in a known target/draft pair (`qwen3-8b`, `qwen3.5-9b`);
`--model`/`--draft` still work for anything else.

```bash
dflash benchmark transformers \
    --model-preset qwen3.5-9b --context-length 16384 \
    --max-samples 32 --max-new-tokens 512 --reasoning off \
    --profile-draft-memory
```

Results are written to `record/<model>_<context-length>_<date>.json`, holding
the full run configuration, a per-sample breakdown, and aggregates for
acceptance rate, acceptance-length histogram, drafter proposal counts, TTFT,
per-decode-token latency, throughput, token counts, and memory. The baseline is
profiled into the same file under its own key, carrying only the metrics that
apply to it — latency, tokens and memory, but nothing drafter-specific. Add
`--no-baseline` to skip that run, which halves the runtime but drops the
speedup number. `--context-task` selects a LongBench-E task (default: every
English task); the paper's long-context tasks are `hotpotqa`, `qasper` and
`gov_report`.

See [PROFILING.md](PROFILING.md) for the file-by-file layout, the record
schema, and per-model run recipes.

Memory is reported as four numbers, because the CUDA allocator cannot attribute
a peak to one of two models sharing a device: overall peak, draft weights,
draft KV cache, and — under `--profile-draft-memory` — the drafter's activation
peak, measured around each draft forward. That flag perturbs latency slightly,
so leave it off when timing is what matters.

## Acknowledgement

Huge thanks to [@dcw02](https://github.com/dcw02), [@gongy](https://github.com/gongy), and the team at [@modal-labs](https://github.com/modal-labs) for their fast, high-quality support in bringing DFlash to SGLang. And huge thanks as well to [@benchislett](https://github.com/benchislett) at NVIDIA for his work in bringing DFlash to vLLM and helping make it available to the broader serving community.

## Citation
If you find DFlash useful, please cite our work. To share feedback on DFlash or request new model support, please fill out this form: [DFlash Feedback](https://forms.gle/4YNwfqb4nJdqn6hq9).

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}

@misc{inco2026dflash2,
  title  = {DFlash 2: Keep Drafting Parallel},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}
```
