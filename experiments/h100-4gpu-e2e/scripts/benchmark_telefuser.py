#!/usr/bin/env python3
"""Measure the complete TeleFuser FP8 Sol MiniMax-H3 profile on four H100s."""

from __future__ import annotations

import argparse
import gc
import platform
import time
from pathlib import Path
from statistics import median

import torch

from benchmark_common import (
    MemorySampler,
    file_sha256,
    git_commit,
    probe_mp4,
    query_memory_mib,
    summarize_times,
    visible_physical_gpus,
    write_report,
)
from examples.minimax_h3.common import save_generation
from examples.minimax_h3.minimax_h3_fl2va_h100 import get_pipeline, run
from telefuser.core.config import AttnImplType


PROMPT = (
    "integrated_multimodal_description: A red fox runs through fresh snow at dawn. "
    "overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds."
)


def ensure_clean_gpus(physical_gpus: list[int], limit_mib: int) -> dict[str, int]:
    memory = query_memory_mib()
    selected = {str(index): memory[index] for index in physical_gpus}
    dirty = {index: used for index, used in selected.items() if used > limit_mib}
    if dirty:
        raise RuntimeError(
            f"refusing to benchmark on occupied GPUs: {dirty}; "
            f"the pre-load limit is {limit_mib} MiB"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telefuser-repo", type=Path, required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-seed", type=int, default=999)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-initial-memory-mib", type=int, default=1024)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    physical_gpus = visible_physical_gpus(4)
    initial_memory = ensure_clean_gpus(physical_gpus, args.max_initial_memory_mib)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    pipeline = get_pipeline(
        4,
        args.model_root,
        num_inference_steps=5,
        enable_fsdp=False,
        online_adaln_cache=True,
        attn_impl=AttnImplType.SOL_ATTN,
        attention_chunks=2,
        ulysses_sequence_mode="valid_only",
        sol_fp8=True,
        sol_dense_steps=1,
        sol_dense_layers=0,
        sol_tau=1.0,
        sol_threshold_type="exact",
        sol_fp8_smoothing="kv",
        sol_fp8_v_bias_correction=True,
        quantization="tf-kernel-fp8",
        adapter_path=str(args.adapter_path),
        adapter_strength=1.0,
    )
    load_seconds = time.perf_counter() - load_started

    samples: list[dict[str, object]] = []
    try:
        warmup_started = time.perf_counter()
        warmup = run(
            pipeline,
            prompt=args.prompt,
            seed=args.warmup_seed,
            aspect_ratio="16:9",
            target_video_length=5.0,
            mode="t2va",
        )
        save_generation(warmup, args.output_dir / "warmup.mp4")
        warmup_seconds = time.perf_counter() - warmup_started
        del warmup
        gc.collect()

        for index in range(1, args.repeats + 1):
            output = args.output_dir / f"run-{index:02d}.mp4"
            sampler = MemorySampler(physical_gpus)
            with sampler:
                e2e_started = time.perf_counter()
                generation_started = e2e_started
                result = run(
                    pipeline,
                    prompt=args.prompt,
                    seed=args.seed,
                    aspect_ratio="16:9",
                    target_video_length=5.0,
                    mode="t2va",
                )
                generation_seconds = time.perf_counter() - generation_started
                save_started = time.perf_counter()
                save_generation(result, output)
                save_seconds = time.perf_counter() - save_started
                e2e_seconds = time.perf_counter() - e2e_started
            validity = probe_mp4(output, width=1344, height=768, frames=124)
            if not validity["valid"]:
                raise RuntimeError(f"invalid generated media: {validity['errors']}")
            samples.append(
                {
                    "index": index,
                    "output": str(output),
                    "e2e_seconds": e2e_seconds,
                    "generation_seconds": generation_seconds,
                    "save_seconds": save_seconds,
                    "denoising_seconds": float(
                        result.runtime_metrics["denoising_seconds"]
                    ),
                    "runtime_metrics": dict(result.runtime_metrics),
                    "memory": sampler.summary(),
                    "validity": validity,
                }
            )
            del result
            gc.collect()
    finally:
        pipeline.stop()

    e2e = [float(sample["e2e_seconds"]) for sample in samples]
    denoise = [float(sample["denoising_seconds"]) for sample in samples]
    peak_per_gpu = max(
        int(sample["memory"]["max_per_gpu_peak_mib"]) for sample in samples
    )
    aggregate_peak = max(
        int(sample["memory"]["aggregate_peak_mib"]) for sample in samples
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "framework": "TeleFuser",
        "profile": "fp8-linear-fp8-sol-smoothing",
        "commit": git_commit(args.telefuser_repo),
        "hardware": "4x NVIDIA H100 80GB HBM3",
        "physical_gpus": physical_gpus,
        "initial_memory_mib": initial_memory,
        "parallelism": {"tp_degree": 2, "ulysses_sp_degree": 2},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": "MiniMaxAI/MiniMax-H3",
        "adapter": {
            "name": "FastH3 Dense/Data-Free",
            "sha256": file_sha256(args.adapter_path),
            "strength": 1.0,
        },
        "workload": {
            "task": "t2va",
            "prompt": args.prompt,
            "seed": args.seed,
            "warmup_seed": args.warmup_seed,
            "resolution": [1344, 768],
            "frames": 124,
            "fps": 24,
            "duration_seconds": 5.0,
            "sigma_points": 5,
            "actual_dit_forwards": 4,
        },
        "configuration": {
            "linear": "tf-kernel FP8 W8A8",
            "attention": "FP8 Sol-Attn",
            "dense_steps": 1,
            "dense_layers": 0,
            "tau": 1.0,
            "threshold": "exact",
            "smoothing": "kv",
            "v_bias_correction": True,
            "attention_chunks": 2,
            "sequence_mode": "valid_only",
            "online_adaln_cache": True,
        },
        "measurement": {
            "scope": "prompt processing through synchronized MP4 close",
            "warmups": 1,
            "repeats": args.repeats,
            "load_seconds": load_seconds,
            "warmup_seconds": warmup_seconds,
            "e2e": summarize_times(e2e),
            "denoising": summarize_times(denoise),
            "videos_per_hour": 3600.0 / median(e2e),
            "actual_dit_forwards_per_second": 4.0 / median(denoise),
            "max_per_gpu_peak_mib": peak_per_gpu,
            "aggregate_peak_mib": aggregate_peak,
        },
        "samples": samples,
    }
    write_report(args.metrics_json, report)


if __name__ == "__main__":
    main()
