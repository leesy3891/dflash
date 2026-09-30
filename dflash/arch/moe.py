"""What a MoE layer routed, and what each part of it cost.

Research question 2 asks for the per-layer verify cost of Qwen3.5-35B-A3B
split into route, dispatch, expert GEMM, mixer and the draft/target boundary.
Three of those are separable by construction in the HuggingFace
implementation and one is not, so this module is careful about which is which.

``Qwen3_5MoeTopKRouter.forward`` is the route: a
``(tokens, hidden) x (hidden, num_experts)`` GEMM, a softmax and a top-k. It
is its own module, so it is timed on its own.

``Qwen3_5MoeExperts.forward`` is dispatch and expert GEMM together. The
reference implementation builds a one-hot mask over experts, finds the hit
set, then loops: gather this expert's tokens, two GEMMs, scatter-add back.
The gather/scatter is dispatch and the two linears are the GEMM, interleaved
per expert in one Python loop. This module times the loop as a whole and
records the token counts that set its shape. It does **not** divide that time
by tokens to produce a per-token cost: with 8 experts per token and 256
experts per layer, the loop's cost is set by how many experts were hit and
how unbalanced the hits were, and a per-token average hides exactly that.

What is recorded instead is the routing itself -- the top-k expert IDs, how
many distinct experts a layer actually touched, the token count per expert,
and the imbalance between them -- so the cost can be explained by the shape
rather than attributed by division.

Capture is off by default and costs a device-to-host copy of the top-k index
tensor per MoE layer per forward, which is why it is a separate pass from the
performance one.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from .taxonomy import FFN_MOE, FFN_MOE_SHARED, decoder_layers, describe


def _is_moe(layer) -> bool:
    mlp = layer._modules.get("mlp")
    return mlp is not None and "experts" in mlp._modules


class RoutingCapture:
    """Hooks every MoE layer of a target and records what it routed.

    ``max_tokens`` bounds how much of a prefill's routing is kept: a 64k
    prefill routes 64k tokens through 40 layers at top-8, which is 20M expert
    IDs and not something to hold in a record. Verify blocks are 16 tokens, so
    decode-time routing is kept whole; prefill is summarised only.
    """

    def __init__(self, target, *, max_tokens: int = 256, time_parts: bool = True):
        self.target = target
        self.max_tokens = max_tokens
        self.time_parts = time_parts
        self.records: list[dict] = []
        self._handles: list = []
        self._spec = describe(target)
        self._moe_layers = {
            entry["index"]: entry
            for entry in self._spec["layers"]
            if entry["ffn"] in (FFN_MOE, FFN_MOE_SHARED)
        }
        self.phase = "unset"
        self.step: int | None = None

    # -- hooks ------------------------------------------------------------
    def _router_hooks(self, index: int, gate):
        state = {}

        def pre(module, args):
            if self.time_parts:
                torch.cuda.synchronize(next(module.parameters()).device)
                state["t0"] = time.perf_counter()

        def post(module, args, output):
            router_logits, scores, indices = output
            elapsed = None
            if self.time_parts and "t0" in state:
                torch.cuda.synchronize(next(module.parameters()).device)
                elapsed = time.perf_counter() - state["t0"]
            self._record_routing(index, indices, scores, elapsed)

        return [
            gate.register_forward_pre_hook(pre),
            gate.register_forward_hook(post),
        ]

    def _expert_hooks(self, index: int, experts):
        state = {}

        def pre(module, args):
            if self.time_parts:
                torch.cuda.synchronize(next(module.parameters()).device)
                state["t0"] = time.perf_counter()

        def post(module, args, output):
            if self.time_parts and "t0" in state:
                torch.cuda.synchronize(next(module.parameters()).device)
                elapsed = time.perf_counter() - state["t0"]
                for record in reversed(self.records):
                    if record["layer"] == index and record.get("expert_loop_s") is None:
                        record["expert_loop_s"] = elapsed
                        break

        return [
            experts.register_forward_pre_hook(pre),
            experts.register_forward_hook(post),
        ]

    def _record_routing(self, index, indices, scores, router_s) -> None:
        num_tokens = indices.shape[0]
        entry = self._moe_layers.get(index, {})
        facts = entry.get("moe", {})
        num_experts = facts.get("num_experts")
        counts = torch.bincount(
            indices.flatten(), minlength=num_experts or 0
        ).to("cpu")
        hit = int((counts > 0).sum().item())
        nonzero = counts[counts > 0].float()
        record = {
            "phase": self.phase,
            "step": self.step,
            "layer": index,
            "module": entry.get("module_path"),
            "device": entry.get("device"),
            "mixer": entry.get("mixer"),
            "num_tokens": num_tokens,
            "top_k": facts.get("top_k"),
            "num_experts": num_experts,
            "unique_experts_hit": hit,
            "expert_hit_fraction": (hit / num_experts) if num_experts else None,
            # Load imbalance over the experts that were actually hit: max
            # tokens on one expert divided by the mean over hit experts. The
            # loop runs once per hit expert, so this is what sets its tail.
            "tokens_per_expert_max": int(nonzero.max().item()) if hit else 0,
            "tokens_per_expert_mean": float(nonzero.mean().item()) if hit else 0.0,
            "load_imbalance": (
                float(nonzero.max().item() / nonzero.mean().item()) if hit else None
            ),
            "router_s": router_s,
            "expert_loop_s": None,
        }
        if num_tokens <= self.max_tokens:
            record["top_k_expert_ids"] = indices.to("cpu").tolist()
            record["routing_weights"] = scores.float().to("cpu").tolist()
        else:
            record["top_k_expert_ids"] = None
            record["na_reason"] = (
                f"per-token expert IDs not kept: {num_tokens} tokens exceeds "
                f"max_tokens={self.max_tokens}"
            )
        record["tokens_per_expert"] = counts.tolist() if num_experts else None
        self.records.append(record)

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "RoutingCapture":
        layers = decoder_layers(self.target)
        for index in self._moe_layers:
            mlp = layers[index]._modules["mlp"]
            gate = mlp._modules.get("gate")
            experts = mlp._modules.get("experts")
            if gate is not None:
                self._handles.extend(self._router_hooks(index, gate))
            if experts is not None and self.time_parts:
                self._handles.extend(self._expert_hooks(index, experts))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def region(self, phase: str, step: int | None = None):
        previous = (self.phase, self.step)
        self.phase, self.step = phase, step
        try:
            yield
        finally:
            self.phase, self.step = previous

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        """Per-phase aggregates, with the timing caveat attached."""
        by_phase: dict = {}
        for record in self.records:
            bucket = by_phase.setdefault(
                record["phase"],
                {
                    "layers": 0, "router_s": 0.0, "expert_loop_s": 0.0,
                    "unique_experts_hit": [], "load_imbalance": [],
                    "tokens": 0, "router_s_missing": 0,
                },
            )
            bucket["layers"] += 1
            bucket["tokens"] += record["num_tokens"]
            if record["router_s"] is not None:
                bucket["router_s"] += record["router_s"]
            else:
                bucket["router_s_missing"] += 1
            if record["expert_loop_s"] is not None:
                bucket["expert_loop_s"] += record["expert_loop_s"]
            bucket["unique_experts_hit"].append(record["unique_experts_hit"])
            if record["load_imbalance"] is not None:
                bucket["load_imbalance"].append(record["load_imbalance"])

        for bucket in by_phase.values():
            hits = bucket.pop("unique_experts_hit")
            imbalance = bucket.pop("load_imbalance")
            bucket["mean_unique_experts_hit"] = (
                sum(hits) / len(hits) if hits else None
            )
            bucket["max_unique_experts_hit"] = max(hits) if hits else None
            bucket["mean_load_imbalance"] = (
                sum(imbalance) / len(imbalance) if imbalance else None
            )
        return {
            "by_phase": by_phase,
            "num_moe_layers": len(self._moe_layers),
            "timing_note": (
                "router_s and expert_loop_s are measured with a device "
                "synchronize on each side, so they are not comparable with "
                "the unprofiled performance pass. expert_loop_s covers "
                "dispatch and expert GEMM together and is deliberately not "
                "divided by token count."
            ),
        }
