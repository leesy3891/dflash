"""NVTX ranges at phase -> layer -> operator, and nothing when profiling is off.

The ranges exist so an Nsight Systems timeline can be read against the same
phase names the memory and performance passes use. They are pushed on the
host, so they mark launch order rather than execution order; correlating them
to the kernels that actually ran is Nsight's job, via the launch correlation
IDs, and is not attempted here.

Enabled only when ``DFLASH_NVTX=1``. Off, every call is a no-op that costs a
module-level boolean check, so the same instrumented code path serves the
performance pass without perturbing it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

ENABLED = os.environ.get("DFLASH_NVTX", "0") == "1"

_push = _pop = _mark = None
BACKEND = None

if ENABLED:
    try:
        import nvtx as _nvtx

        _push = lambda label: _nvtx.push_range(label)  # noqa: E731
        _pop = lambda: _nvtx.pop_range()  # noqa: E731
        _mark = lambda label: _nvtx.mark(label)  # noqa: E731
        BACKEND = "nvtx"
    except ImportError:
        try:
            import torch.cuda.nvtx as _tnvtx

            _push = _tnvtx.range_push
            _pop = _tnvtx.range_pop
            _mark = _tnvtx.mark
            BACKEND = "torch.cuda.nvtx"
        except ImportError:  # pragma: no cover
            ENABLED = False


@contextmanager
def range(label: str, **tags):
    """One NVTX range. Tags are appended to the label, which is all Nsight sees."""
    if not ENABLED:
        yield
        return
    if tags:
        suffix = " ".join(f"{k}={v}" for k, v in tags.items() if v is not None)
        label = f"{label} [{suffix}]" if suffix else label
    _push(label)
    try:
        yield
    finally:
        _pop()


def mark(label: str) -> None:
    if ENABLED:
        _mark(label)


def phase(name: str, **tags):
    return range(f"phase/{name}", **tags)


def layer(role: str, index: int, mixer: str | None = None, ffn: str | None = None):
    return range(f"layer/{role}/{index}", mixer=mixer, ffn=ffn)


def operator(name: str, **tags):
    return range(f"op/{name}", **tags)


def status() -> dict:
    return {"enabled": ENABLED, "backend": BACKEND}
