# -*- coding: utf-8 -*-
"""Device auto-detection: use CUDA when available, fall back to CPU.

Env overrides:
  EARLOG_DEVICE=cuda:0   force a specific device (default: auto)
  EARLOG_DEVICE=cpu      force CPU even when a GPU is present
  EARLOG_SPK_DEVICE=...  speaker stream only (default: cpu, see speaker_device)
"""
import os

import torch


def get_device():
    forced = os.environ.get("EARLOG_DEVICE", "").strip()
    if forced:
        return forced
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def on_gpu(device=None):
    return (device or get_device()).startswith("cuda")


def funasr_device(device=None):
    """funasr AutoModel accepts 'cpu' / 'cuda:0' directly."""
    return device or get_device()


def speaker_device(device=None):
    """CAM++ runs on CPU by default even when a GPU is present.

    Measured 2026-10-21 on RTX 3050 Laptop 4GB: GPU is only 1.4x faster
    (78s -> 54s per 30min audio) but the process dies silently in ~60% of
    runs -- no traceback, no output file, just a CUDA assert
    (TORCH_USE_CUDA_DSA hint). Not worth the risk for a 1.4x gain.
    Opt in with EARLOG_SPK_DEVICE=cuda:0 only if you add retry logic.
    """
    forced = os.environ.get("EARLOG_SPK_DEVICE", "").strip()
    if forced:
        return forced
    return "cpu"


def panns_device(device=None):
    """panns_inference expects 'cpu' or 'cuda' (no index suffix)."""
    return "cuda" if on_gpu(device) else "cpu"


def describe():
    if on_gpu():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"{get_device()} ({name}, {vram:.1f}GB VRAM)"
    return f"cpu ({os.cpu_count()} threads)"
