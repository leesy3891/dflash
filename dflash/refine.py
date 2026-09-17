"""Training-free local causal refinement of one DFlash draft block.

DFlash drafts a whole block in one forward: every masked position is predicted
from the same MASK embedding, so no position sees what the drafter proposes at
the positions before it. This module adds a weak causal signal *after* the
single drafter forward and the single full-vocabulary LM-head call, without
repeating either:

    draft forward -> full LM head -> top-K soft token
                  -> local causal refinement -> top-K rerank -> target verify

1. ``refine_topk_soft_embedding``: top-K of each position's draft logits,
   renormalised, gives an expected embedding ``e_soft[t] = sum_v p_t(v) E[v]``
   over the target's input embedding table, and ``delta_e[t] = e_soft[t] -
   e_mask`` is how far that position's belief moved away from MASK.
2. ``refine_kv_projection``: the last draft layer's input hidden is patched,
   ``h'[t] = h[t] + alpha * delta_e[t]``, and only the block-local K/V are
   re-projected from it (input norm, k/v proj, k norm, RoPE). Q is the one the
   drafter already computed for that layer, captured during its forward.
3. ``refine_attention``: queries at positions 1..L-1 attend to the previous
   ``window`` block positions only (strictly earlier, never the context and
   never the draft KV cache), once with the patched K/V and once with the
   original ones captured from the forward. The difference, through
   ``o_proj``, is the change in that layer's attention output that the
   neighbours' soft tokens cause.
4. ``refine_candidate_rerank``: that change is added to the pre-norm final
   hidden, both hiddens go through the final norm, and the difference is dotted
   with the LM-head rows of the K candidates only. ``score = logit + delta``;
   the argmax over K is the proposal.

The last layer's MLP is not re-run on the perturbation (first-order: the
change reaches the output through the residual path only), so the only
matrix multiplications against the vocabulary are ``T x K`` rows, never ``T x
V``. Nothing here writes to a cache: every tensor is local to one call and
freed when it returns, so ``past_key_values_draft`` and ``_crop_to`` are
untouched. Greedy verification keeps the output exact whatever is proposed.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.nn import functional as F

STAGES = (
    "refine_topk_soft_embedding",
    "refine_kv_projection",
    "refine_attention",
    "refine_candidate_rerank",
)


@dataclass(frozen=True)
class RefineConfig:
    top_k: int = 16
    # Previous block positions each query may attend to; None is the whole
    # prefix of the block.
    window: int | None = 1
    alpha: float = 1.0

    @property
    def window_label(self) -> str:
        return "full" if self.window is None else str(self.window)

    @property
    def name(self) -> str:
        alpha = "" if self.alpha == 1.0 else f"_a{self.alpha:g}"
        return f"refine_w{self.window_label}_k{self.top_k}{alpha}"

    def as_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "window": self.window_label,
            "alpha": self.alpha,
        }


def parse_window(value: str) -> int | None:
    value = str(value).strip().lower()
    if value == "full":
        return None
    window = int(value)
    if window < 1:
        raise ValueError("--refine-window must be a positive integer or 'full'")
    return window


def local_causal_mask(length: int, window: int | None, device) -> torch.Tensor:
    """Visibility of key ``s`` to query ``t'`` after the shift by one.

    Queries are block positions 1..L-1 and keys 0..L-2, so row ``t'`` is block
    position ``t'+1`` and column ``s`` is block position ``s``. Strictly earlier
    means ``s <= t'``; the window keeps ``(t'+1) - s <= window``. Every row has
    at least its immediate predecessor, so no row is fully masked.
    """
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    visible = key <= query
    if window is not None:
        visible &= query - key < window
    return visible


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def capture_for(model) -> dict:
    """An empty capture that the drafter's last layer fills during its forward."""
    return {"layer_idx": len(model.layers) - 1}


def local_refine(
    model,
    target,
    capture: dict,
    draft_logits: torch.Tensor,
    config: RefineConfig,
    *,
    mask_token_id: int,
    embedding_scale: float,
    output_head: torch.nn.Module,
    output_multiplier: float = 1.0,
    stage=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rerank the top-K draft candidates of one block. Greedy only.

    ``draft_logits`` is ``(1, L-1, V)`` for block positions 1..L-1, and
    ``capture`` holds what the drafter's last layer saw for all L positions.
    Returns the proposed tokens ``(1, L-1)`` and the chosen candidate rank per
    position (0 means the original argmax was kept).
    """
    stage = stage or (lambda name: nullcontext())
    attention = model.layers[-1].self_attn
    layer = model.layers[-1]
    head_dim = attention.head_dim
    q = capture["q"]  # (1, Hq, L, d), RoPE applied
    k_old = capture["k"]  # (1, Hkv, L, d), k_norm + RoPE applied
    v_old = capture["v"]  # (1, Hkv, L, d)
    layer_input = capture["layer_input"]  # (1, L, H), pre input_layernorm
    prenorm = capture["prenorm"]  # (1, L, H), pre final norm
    length = layer_input.shape[1]
    device, dtype = layer_input.device, layer_input.dtype
    num_draft = length - 1

    with stage("refine_topk_soft_embedding"):
        top_k = min(config.top_k, draft_logits.shape[-1])
        top_values, top_indices = torch.topk(draft_logits, top_k, dim=-1)
        probs = torch.softmax(top_values.float(), dim=-1).to(dtype)
        embedding = target.get_input_embeddings().weight
        candidate_embedding = F.embedding(
            top_indices.to(embedding.device), embedding
        ).to(device)  # (1, L-1, K, H)
        soft = torch.einsum("btk,btkh->bth", probs, candidate_embedding)
        mask_embedding = embedding[mask_token_id].to(device)
        delta_e = (soft - mask_embedding) * embedding_scale
        del candidate_embedding, soft

    # Keys are block positions 0..L-2: position 0 is the verified anchor, whose
    # input carries no MASK and is not patched, so its K/V are reused as is.
    # Position L-1 is nobody's predecessor and is never projected.
    with stage("refine_kv_projection"):
        patched_len = length - 2
        if patched_len > 0:
            patched = layer_input[:, 1:-1] + config.alpha * delta_e[:, :-1]
            patched = layer.input_layernorm(patched)
            k_new = attention.k_proj(patched).view(1, patched_len, -1, head_dim)
            k_new = attention.k_norm(k_new).transpose(1, 2)
            v_new = (
                attention.v_proj(patched)
                .view(1, patched_len, -1, head_dim)
                .transpose(1, 2)
            )
            cos = capture["cos"][:, None, 1:-1]
            sin = capture["sin"][:, None, 1:-1]
            k_new = k_new * cos + _rotate_half(k_new) * sin
            k_new = torch.cat([k_old[:, :, :1], k_new], dim=2)
            v_new = torch.cat([v_old[:, :, :1], v_new], dim=2)
            del patched
        else:
            k_new, v_new = k_old[:, :, :1], v_old[:, :, :1]

    with stage("refine_attention"):
        mask = local_causal_mask(num_draft, config.window, device)
        queries = q[:, :, 1:].expand(2, -1, -1, -1)
        keys = torch.cat([k_new, k_old[:, :, :-1]], dim=0)
        values = torch.cat([v_new, v_old[:, :, :-1]], dim=0)
        both = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
            scale=attention.scaling,
            enable_gqa=True,
        )
        delta_attention = attention.o_proj(
            (both[:1] - both[1:]).transpose(1, 2).reshape(1, num_draft, -1)
        )
        del queries, keys, values, both, k_new, v_new

    with stage("refine_candidate_rerank"):
        hidden = prenorm[:, 1:]
        delta_hidden = (
            model.norm(hidden + delta_attention) - model.norm(hidden)
        ).float()
        rows = F.embedding(
            top_indices.to(output_head.weight.device), output_head.weight
        ).to(device)  # (1, L-1, K, H): the candidates' rows only
        delta_logit = torch.einsum("btkh,bth->btk", rows.float(), delta_hidden)
        scores = top_values.float() + delta_logit * output_multiplier
        rank = torch.argmax(scores, dim=-1)
        tokens = top_indices.gather(-1, rank[..., None])[..., 0]
        del rows, delta_hidden, delta_logit, scores

    capture.clear()
    capture["layer_idx"] = len(model.layers) - 1
    return tokens, rank
