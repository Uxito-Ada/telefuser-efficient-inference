#!/usr/bin/env python3
"""Shared, framework-neutral measurement helpers for the H3 benchmark."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path
from statistics import median
from typing import Any


def visible_physical_gpus(expected: int) -> list[int]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        devices = list(range(expected))
    if len(devices) != expected:
        raise ValueError(
            f"expected exactly {expected} visible GPUs, found {devices}; "
            "the benchmark does not admit a different GPU count"
        )
    return devices


def query_memory_mib() -> dict[int, int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        int(index.strip()): int(memory.strip())
        for index, memory in (line.split(",") for line in output.splitlines())
    }


class MemorySampler:
    def __init__(self, physical_gpus: list[int], interval_seconds: float = 0.1):
        self.physical_gpus = physical_gpus
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def _sample(self) -> None:
        while not self._stop.is_set():
            values = query_memory_mib()
            selected = {str(index): values[index] for index in self.physical_gpus}
            self.samples.append(
                {
                    "elapsed_seconds": time.perf_counter() - self._started,
                    "memory_mib": selected,
                    "aggregate_mib": sum(selected.values()),
                }
            )
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> MemorySampler:
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            raise RuntimeError("memory sampler recorded no samples")
        per_gpu = {
            str(index): max(row["memory_mib"][str(index)] for row in self.samples)
            for index in self.physical_gpus
        }
        return {
            "sample_interval_seconds": self.interval_seconds,
            "per_gpu_peak_mib": per_gpu,
            "max_per_gpu_peak_mib": max(per_gpu.values()),
            "aggregate_peak_mib": max(row["aggregate_mib"] for row in self.samples),
            "samples": self.samples,
        }


def probe_mp4(path: Path, *, width: int, height: int, frames: int) -> dict[str, Any]:
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,codec_type,width,height,r_frame_rate,nb_read_frames,channels,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    probe = json.loads(output)
    streams = probe.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), None
    )
    errors: list[str] = []
    if video is None:
        errors.append("missing video stream")
    else:
        if [video.get("width"), video.get("height")] != [width, height]:
            errors.append(
                f"unexpected dimensions: {video.get('width')}x{video.get('height')}"
            )
        if int(video.get("nb_read_frames", -1)) != frames:
            errors.append(f"unexpected frame count: {video.get('nb_read_frames')}")
        numerator, denominator = (
            int(part) for part in str(video.get("r_frame_rate", "0/1")).split("/")
        )
        frame_rate = numerator / denominator
        if abs(frame_rate - 24.0) > 1e-6:
            errors.append(f"unexpected frame rate: {frame_rate}")
    if audio is None:
        errors.append("missing audio stream")
    elif int(audio.get("channels", 0)) != 2:
        errors.append(f"unexpected audio channels: {audio.get('channels')}")
    if int(probe.get("format", {}).get("size", 0)) <= 1024:
        errors.append("output file is empty or implausibly small")
    return {"path": str(path), "valid": not errors, "errors": errors, "ffprobe": probe}


def git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_times(values: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(values)
    center = median(values)
    deviations = [abs(value - center) for value in values]
    return {
        "samples": values,
        "median": center,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "median_absolute_deviation": median(deviations),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
