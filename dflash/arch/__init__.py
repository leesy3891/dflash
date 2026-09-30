"""B=1 architecture profiling for the main branch.

This package is the measurement layer for the three main-branch research
questions: where a hybrid target's decode state goes when its attention KV
shrinks, what a MoE layer's verify actually costs, and how the first-draft
setup / steady draft / verify split moves with S and block width.

It is deliberately separate from :mod:`dflash.benchmark`, which stays as the
record of the earlier sweeps. Nothing here is backward compatible with those
records; new results are written under ``record_arch_main/``.

Scope: one request at a time. No static batching, no per-row KV, no
continuous scheduling -- those belong to the batch branch. The four GPUs here
carry a layer-sharded target, which is one request executing across four
devices, not four requests.
"""

__all__ = [
    "env",
    "events",
    "placement",
    "statemem",
    "taxonomy",
]
