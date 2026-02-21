"""
Utility functions for FairMOP.

Provides hardware detection (GPU), device management, and
miscellaneous helpers used across the framework.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional

# ── GPU Detection ────────────────────────────────────────────────────────────


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get information about available NVIDIA GPUs.

    Tries ``nvidia-smi`` first, falls back to PyTorch detection.

    Returns:
        List of GPU info dictionaries with keys:
        ``index``, ``name``, ``memory_used``, ``memory_total``,
        ``memory_free``, ``utilization``, ``memory_usage_percent``.
    """
    try:
        return _get_gpu_info_nvidia_smi()
    except Exception:
        return _get_gpu_info_torch()


def _get_gpu_info_nvidia_smi() -> List[Dict[str, Any]]:
    """Detect GPUs using nvidia-smi."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode != 0:
        raise RuntimeError("nvidia-smi failed")

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                mem_used = int(parts[2])
                mem_total = int(parts[3])
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used": mem_used,
                        "memory_total": mem_total,
                        "memory_free": mem_total - mem_used,
                        "utilization": int(parts[4]),
                        "memory_usage_percent": (mem_used / mem_total) * 100
                        if mem_total > 0
                        else 0,
                    }
                )
            except (ValueError, ZeroDivisionError):
                continue

    return gpus


def _get_gpu_info_torch() -> List[Dict[str, Any]]:
    """Detect GPUs using PyTorch as fallback."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mb = props.total_memory // (1024**2)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "memory_total": total_mb,
                    "memory_used": 0,
                    "memory_free": total_mb,
                    "utilization": 0,
                    "memory_usage_percent": 0,
                }
            )
        return gpus

    except Exception:
        return []


def set_gpu_device(gpu_index: Optional[int]) -> str:
    """Set the PyTorch CUDA device.

    Parameters:
        gpu_index: GPU device index, or None for CPU.

    Returns:
        Device string (e.g., ``"cuda:0"`` or ``"cpu"``).
    """
    try:
        import torch

        if torch.cuda.is_available() and gpu_index is not None:
            if gpu_index < torch.cuda.device_count():
                torch.cuda.set_device(gpu_index)
                print(
                    f"[FairMOP] Using GPU {gpu_index}: "
                    f"{torch.cuda.get_device_name(gpu_index)}"
                )
                return f"cuda:{gpu_index}"
            else:
                print(
                    f"[FairMOP] GPU {gpu_index} not available. "
                    f"Using default CUDA device."
                )
                return "cuda"

        return "cpu"

    except ImportError:
        print("[FairMOP] PyTorch not available. Using CPU.")
        return "cpu"


def select_best_gpu() -> Optional[int]:
    """Select the GPU with the most free memory.

    Returns:
        GPU index, or None if no GPU is available.
    """
    gpus = get_gpu_info()
    if not gpus:
        return None
    best = max(gpus, key=lambda g: g["memory_free"])
    return best["index"]


# ── File helpers ─────────────────────────────────────────────────────────────


def count_images(directory: str) -> int:
    """Count image files in a directory (recursively).

    Parameters:
        directory: Path to the directory.

    Returns:
        Number of image files found.
    """
    import glob

    count = 0
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        count += len(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return count


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist.

    Parameters:
        path: Directory path.

    Returns:
        The directory path.
    """
    os.makedirs(path, exist_ok=True)
    return path
