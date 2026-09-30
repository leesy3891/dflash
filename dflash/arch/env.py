"""What the machine and the software stack were, recorded once per run.

Every number in a record is only comparable against another number taken on
the same stack, so the manifest is written next to the results rather than
kept in a lab notebook. The GPU entries carry UUIDs because "GPU 0" is a
per-process label -- ``CUDA_VISIBLE_DEVICES`` renames cards, and a sweep that
moves between them silently is a sweep whose placement is not fixed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

SCHEMA_VERSION = "arch-main/2"


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state(repo: str | None = None) -> dict:
    """The commit the code was at, and whether the tree was dirty.

    A dirty tree is recorded rather than refused: a profiling run is often the
    thing that motivates the next edit. But a record whose SHA does not
    reproduce its own code is worth saying so out loud.
    """
    cwd = repo or os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    head = _run(["git", "-C", cwd, "rev-parse", "HEAD"])
    branch = _run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "-C", cwd, "status", "--porcelain"])
    return {
        "commit": head,
        "branch": branch,
        "dirty": bool(status),
        "dirty_paths": (status.splitlines()[:50] if status else []),
    }


def package_versions() -> dict:
    """Versions of everything whose kernels can change a measured number."""
    versions: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "fla",
        "flash_attn",
        "causal_conv1d",
        "triton",
        "nvtx",
        "safetensors",
    ):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - absence is the datum
            versions[name] = None
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def _nvidia_smi_topology() -> str | None:
    """The P2P/NVLink matrix, if this driver build reports one."""
    for args in (["topo", "-m"], ["topo", "-mp"], ["topo", "--matrix"]):
        out = _run(["nvidia-smi", *args])
        if out:
            return out
    return None


def gpu_inventory() -> dict:
    """Every visible card: identity, memory in bytes, and how they connect.

    Memory is reported in bytes rather than the GiB a spec sheet quotes,
    because the capacity questions in this sweep -- what fits at 64k, what
    OOMs -- are decided by the actual figure the allocator sees.
    """
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "count": 0, "devices": [], "topology": None}
    # is_available() does not create a context, and get_device_properties on an
    # uninitialised runtime reports every index as invalid.
    torch.cuda.init()

    devices = []
    # Joined on UUID, never on index. nvidia-smi enumerates by PCI bus while
    # CUDA defaults to FASTEST_FIRST, so the two orderings disagree whenever a
    # card drops out -- and a PCI address attached to the wrong device is worse
    # than none, because it is the field a reader uses to identify the card.
    by_uuid: dict[str, dict] = {}
    for line in (_run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
         "--format=csv,noheader"]
    ) or "").splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit():
            uuid = parts[1].removeprefix("GPU-").lower()
            by_uuid[uuid] = {
                "smi_index": int(parts[0]),
                "smi_uuid": parts[1],
                "smi_name": parts[2],
                "smi_memory_total": parts[3],
                "pci_bus_id": parts[4],
            }

    for index in range(torch.cuda.device_count()):
        # A card the driver has lost (nvidia-smi shows ERR! against it) still
        # counts, so record the failure against that index rather than losing
        # the whole inventory to it.
        try:
            props = torch.cuda.get_device_properties(index)
            entry = {
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "capability": f"{props.major}.{props.minor}",
                "usable": True,
            }
            uuid = str(getattr(props, "uuid", "")).removeprefix("GPU-").lower()
            entry["uuid"] = uuid or None
            entry.update(by_uuid.get(uuid, {}))
        except Exception as exc:  # noqa: BLE001 - an unusable card is the datum
            entry = {"index": index, "usable": False, "error": repr(exc)}
        devices.append(entry)

    matched = {d.get("uuid") for d in devices if d.get("uuid")}
    unmatched_smi = [row for uuid, row in by_uuid.items() if uuid not in matched]

    def _peer(a: int, b: int):
        if a == b:
            return None
        try:
            return bool(torch.cuda.can_device_access_peer(a, b))
        except Exception:  # noqa: BLE001
            return None

    open_indices = [d["index"] for d in devices if d.get("usable")]
    peer = {
        f"{a}->{b}": _peer(a, b)
        for a in open_indices
        for b in open_indices
        if a != b
    }
    return {
        "available": True,
        "count": torch.cuda.device_count(),
        "devices": devices,
        "usable_count": sum(1 for d in devices if d.get("usable")),
        "peer_access": peer,
        "topology": _nvidia_smi_topology(),
        # Cards nvidia-smi lists that CUDA never opened. On a shared box this
        # is how a driver-level failure becomes a line in the record rather
        # than a silently smaller sweep.
        "visible_to_smi_but_not_cuda": unmatched_smi,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST"),
        "driver_cuda": torch.version.cuda,
    }


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def token_hash(token_ids) -> str:
    """A stable fingerprint of a prompt's token IDs.

    Two runs that claim the same context length are only comparable if they
    ran the same tokens; the hash is what lets a later reader check that
    without shipping the prompt.
    """
    import torch

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.flatten().tolist()
    payload = ",".join(str(int(t)) for t in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def manifest(extra: dict | None = None) -> dict:
    """The full run manifest. Written once, at the head of every record."""
    import torch

    data = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git": git_state(),
        "versions": package_versions(),
        "gpu": gpu_inventory(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TOKENIZERS_PARALLELISM",
                "CUDA_LAUNCH_BLOCKING",
            )
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        },
    }
    if extra:
        data.update(extra)
    return data


def write_manifest(path, extra: dict | None = None) -> dict:
    data = manifest(extra)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return data


def usable_devices() -> list[int]:
    """Device indices CUDA can actually open, in the current ordering.

    ``torch.cuda.device_count()`` reports what the driver enumerates, which is
    not the same as what can be used: a card the driver has partially lost
    still raises the count but fails every property query. On the machine this
    sweep was written for that is one of the four A6000s, so a run that trusted
    the count would place layers on a device that cannot hold them.
    """
    import torch

    if not torch.cuda.is_available():
        return []
    torch.cuda.init()
    usable = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except Exception:  # noqa: BLE001
            continue
        usable.append(index)
    return usable


def device_health() -> dict:
    """Enumerated vs usable devices, and which ones the driver has lost."""
    import torch

    enumerated = torch.cuda.device_count() if torch.cuda.is_available() else 0
    usable = usable_devices()
    return {
        "enumerated": enumerated,
        "usable": usable,
        "num_usable": len(usable),
        "unusable": [i for i in range(enumerated) if i not in usable],
        "degraded": len(usable) != enumerated,
    }


# ---------------------------------------------------------------------------
# Fixing the device count
# ---------------------------------------------------------------------------

# A card counts as idle when this much or less is in use and it runs no
# compute process. The box is shared, and a card someone else is using both
# steals memory from a capacity measurement and adds contention to a timing.
IDLE_MEMORY_BYTES = 1 << 30


def _nvml_cards() -> list[dict]:
    """Every card NVML can see, with the facts selection needs.

    Read through NVML rather than CUDA because it has to happen before the CUDA
    runtime is initialised: ``CUDA_VISIBLE_DEVICES`` is read once, at init, and
    setting it afterwards does nothing.
    """
    cards: list[dict] = []
    try:
        import pynvml
    except ImportError:
        pynvml = None
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
        except Exception:  # noqa: BLE001
            pynvml = None
    if pynvml is not None:
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                entry: dict = {"smi_index": index}
                try:
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    entry["uuid"] = uuid.decode() if isinstance(uuid, bytes) else uuid
                    pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
                    entry["pci_bus_id"] = pci.decode() if isinstance(pci, bytes) else pci
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    entry["memory_used_bytes"] = int(memory.used)
                    entry["memory_total_bytes"] = int(memory.total)
                    entry["compute_processes"] = len(
                        pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    )
                    # The lost card on this box answers the memory query but
                    # not this one (nvidia-smi prints [N/A]); CUDA cannot open it.
                    pynvml.nvmlDeviceGetUtilizationRates(handle)
                    entry["healthy"] = True
                except Exception as exc:  # noqa: BLE001 - an unhealthy card is a datum
                    entry["healthy"] = False
                    entry["error"] = repr(exc)
                cards.append(entry)
        finally:
            pynvml.nvmlShutdown()
        return cards

    out = _run([
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]) or ""
    for line in out.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        healthy = parts[5].isdigit()
        cards.append({
            "smi_index": int(parts[0]),
            "uuid": parts[1],
            "pci_bus_id": parts[2],
            "memory_used_bytes": int(parts[3]) << 20 if parts[3].isdigit() else None,
            "memory_total_bytes": int(parts[4]) << 20 if parts[4].isdigit() else None,
            # nvidia-smi cannot list processes per card in this query.
            "compute_processes": None,
            "healthy": healthy,
        })
    return cards


def pin_devices(num_gpus: int, *, uuids: list[str] | None = None,
                allow_busy: bool = False) -> dict:
    """Make exactly ``num_gpus`` idle, healthy cards visible, by UUID.

    Must run before anything initialises CUDA. After it, ``cuda:0`` ..
    ``cuda:{num_gpus-1}`` are the chosen cards in the order chosen, and every
    ``range(torch.cuda.device_count())`` loop in the profiler -- the peak
    trackers, the AR peak read-back -- covers exactly those cards and no
    others. Selecting by UUID rather than index is what keeps this correct on
    a box whose CUDA and nvidia-smi orderings disagree.

    If ``CUDA_VISIBLE_DEVICES`` is already set it is respected: the caller
    has chosen, and this only checks the count after CUDA comes up.
    """
    if num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if preset is not None and not uuids:
        return {
            "num_gpus": num_gpus,
            "mode": "preset_cuda_visible_devices",
            "cuda_visible_devices": preset,
        }

    cards = _nvml_cards()
    wanted = [u if u.startswith("GPU-") else f"GPU-{u}" for u in (uuids or [])]
    if wanted:
        by_uuid = {card.get("uuid"): card for card in cards}
        missing = [u for u in wanted if u not in by_uuid]
        if missing:
            raise RuntimeError(f"requested GPUs not found by NVML: {missing}")
        chosen = [by_uuid[u] for u in wanted]
        if len(chosen) != num_gpus:
            raise RuntimeError(
                f"--gpu-uuids lists {len(chosen)} cards but --num-gpus is {num_gpus}"
            )
    else:
        def idle(card):
            return (
                card.get("memory_used_bytes") is not None
                and card["memory_used_bytes"] <= IDLE_MEMORY_BYTES
                and not card.get("compute_processes")
            )
        candidates = [
            card for card in cards
            if card.get("healthy") and (allow_busy or idle(card))
        ]
        if len(candidates) < num_gpus:
            raise RuntimeError(
                f"need {num_gpus} idle healthy GPUs, found {len(candidates)}: "
                + json.dumps(cards, default=str)
            )
        chosen = candidates[:num_gpus]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(card["uuid"] for card in chosen)
    return {
        "num_gpus": num_gpus,
        "mode": "explicit_uuids" if wanted else "auto_idle",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "chosen": chosen,
        "all_cards_at_selection": cards,
        "idle_memory_threshold_bytes": IDLE_MEMORY_BYTES,
    }


def check_pinned(selection: dict) -> dict:
    """After CUDA is up: the pinned count must be what is usable."""
    usable = usable_devices()
    expected = selection["num_gpus"]
    if len(usable) < expected:
        raise RuntimeError(
            f"pinned {expected} GPUs but CUDA can open {len(usable)}: {usable}"
        )
    selection = dict(selection)
    selection["devices"] = usable[:expected]
    selection["cuda_device_count"] = len(usable)
    if len(usable) > expected:
        # Only reachable with a preset CUDA_VISIBLE_DEVICES wider than the
        # requested count; peak loops over device_count() then include cards
        # the run does not use, which only ever read zero.
        selection["warning"] = (
            f"{len(usable)} devices visible, using the first {expected}"
        )
    return selection
