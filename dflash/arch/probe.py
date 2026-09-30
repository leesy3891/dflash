"""The component split handed to ``dflash_generate`` at every phase boundary.

``PhaseMemory`` calls this once as each phase closes, so every field is read
at the same instant and the phase record says what was simultaneously live --
which is the only way to answer "was the prefill peak the attention KV, the
GDN state, the conv recording buffer, or the hidden materialisation?".

Each phase costs one pass over the cache layers. That is cheap relative to a
target forward but not free, so the probe is only installed on the memory
pass; the performance pass runs without it and is where reported latency
comes from.
"""

from __future__ import annotations

from . import statemem


class ComponentProbe:
    """Reads one resolved component split per call.

    ``target_mixers`` and ``draft_mixers`` are the per-layer mixer axes from
    :func:`taxonomy.describe`, so a hybrid target's layers are labelled by
    what they are rather than by which cache slots happen to be populated.
    """

    def __init__(
        self,
        *,
        target_mixers: list[str] | None = None,
        draft_mixers: list[str] | None = None,
        per_layer: bool = False,
    ) -> None:
        self.target_mixers = target_mixers
        self.draft_mixers = draft_mixers
        self.per_layer = per_layer
        self.calls = 0

    def __call__(self, *, target_cache, draft_cache, live, draft_weight_bytes) -> dict:
        self.calls += 1
        target = statemem.cache_components(target_cache, mixers=self.target_mixers)
        draft = statemem.cache_components(draft_cache, mixers=self.draft_mixers)
        loose = statemem.tensor_components(
            {
                "target_selected_hidden": live.get("selected_hidden") or (),
                "context_feature": (
                    () if live.get("context_feature") is None
                    else [live["context_feature"]]
                ),
            }
        )

        split = {
            # ---- target decode state, resolved -----------------------------
            "target_attention_kv_bytes":
                target["total"][statemem.ATTENTION_KV]["storage_bytes"],
            "target_gdn_recurrent_bytes":
                target["total"][statemem.GDN_RECURRENT]["storage_bytes"],
            "target_gdn_conv_working_bytes":
                target["total"][statemem.GDN_CONV_WORKING]["storage_bytes"],
            # Rollback machinery, not decode state. O(S) during prefill.
            "target_gdn_conv_recording_bytes":
                target["total"][statemem.GDN_CONV_RECORDING]["storage_bytes"],
            "target_cache_unclassified_bytes":
                target["total"][statemem.UNCLASSIFIED]["storage_bytes"],
            # ---- drafter's own state ---------------------------------------
            "draft_attention_kv_bytes":
                draft["total"][statemem.ATTENTION_KV]["storage_bytes"],
            "draft_cache_other_bytes": (
                draft["sum_storage_bytes"]
                - draft["total"][statemem.ATTENTION_KV]["storage_bytes"]
            ),
            "draft_weight_bytes": draft_weight_bytes,
            # ---- what crosses between them ---------------------------------
            "target_selected_hidden_bytes":
                loose["detail"]["target_selected_hidden"]["storage_bytes"],
            "context_feature_bytes":
                loose["detail"]["context_feature"]["storage_bytes"],
            # ---- totals ------------------------------------------------------
            "target_cache_sum_bytes": target["sum_storage_bytes"],
            "draft_cache_sum_bytes": draft["sum_storage_bytes"],
            "allocated_bytes": sum(
                (state or {}).get("allocated_current", 0)
                for state in statemem.device_allocator_state().values()
            ),
        }
        split["per_device"] = {
            "target_cache": {
                device: {k: v["storage_bytes"] for k, v in totals.items()}
                for device, totals in target["per_device"].items()
            },
            "draft_cache": {
                device: {k: v["storage_bytes"] for k, v in totals.items()}
                for device, totals in draft["per_device"].items()
            },
            "loose": loose["per_device"],
            "allocator": statemem.device_allocator_state(),
        }
        if target["aliases"] or draft["aliases"] or loose["aliases"]:
            split["aliases"] = (
                target["aliases"] + draft["aliases"] + loose["aliases"]
            )
        if self.per_layer:
            split["target_per_layer"] = target["per_layer"]
        return split
