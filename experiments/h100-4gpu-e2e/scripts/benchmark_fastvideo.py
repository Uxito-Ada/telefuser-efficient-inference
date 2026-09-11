#!/usr/bin/env python3
"""Measure FastVideo's official dense FastH3 LoRA profile on H100s."""

from __future__ import annotations

import argparse
import gc
import importlib.machinery
import os
import platform
import sys
import time
import types
from pathlib import Path
from statistics import median

import torch

# Editable environments can install an unrelated top-level "examples" package.
# Pin it to the same source tree as the FastVideo package under test.
_source_root = os.environ.get("FASTVIDEO_SOURCE_ROOT")
if _source_root:
    examples = types.ModuleType("examples")
    examples.__path__ = [str(Path(_source_root).resolve() / "examples")]
    examples.__package__ = "examples"
    examples.__spec__ = importlib.machinery.ModuleSpec(
        "examples", loader=None, is_package=True
    )
    sys.modules["examples"] = examples

from benchmark_common import (  # noqa: E402
    MemorySampler,
    file_sha256,
    git_commit,
    probe_mp4,
    query_memory_mib,
    summarize_times,
    visible_physical_gpus,
    write_report,
)
from examples.inference.basic import basic_fasth3  # noqa: E402
from examples.inference.basic import basic_fasth3_lora_preview  # noqa: E402
from fastvideo import VideoGenerator  # noqa: E402


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
    parser.add_argument("--fastvideo-repo", type=Path, required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, choices=(2, 4), default=4)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-seed", type=int, default=999)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-initial-memory-mib", type=int, default=1024)
    outer = parser.parse_args()

    if outer.repeats < 1:
        parser.error("--repeats must be positive")
    physical_gpus = visible_physical_gpus(outer.num_gpus)
    initial_memory = ensure_clean_gpus(physical_gpus, outer.max_initial_memory_mib)
    outer.output_dir.mkdir(parents=True, exist_ok=True)

    official_args = basic_fasth3_lora_preview.parse_args(
        [
            "--model-path",
            outer.model_root,
            "--lora-path",
            str(outer.adapter_path),
            "--lora-strength",
            "1.0",
            "--prompt",
            outer.prompt,
            "--output",
            str(outer.output_dir),
            "--profile",
            "all",
            "--height",
            "768",
            "--width",
            "1344",
            "--num-frames",
            str(outer.num_frames),
            "--steps",
            "5",
            "--seed",
            str(outer.seed),
            "--warmup-seed",
            str(outer.warmup_seed),
            "--repeats",
            str(outer.repeats),
            "--num-gpus",
            str(outer.num_gpus),
            "--lazy-module-load",
            "--no-vsa",
            "--fa4",
            "--h3-fusions",
            "--no-compile-vae",
            "--parallel-vae",
            "--no-replicated-dit",
            "--pin-cpu-memory",
            "--no-inference-torch-compile",
            "--ulysses-a2a",
            "auto",
        ]
    )
    environment = basic_fasth3.configure_environment(official_args)
    basic_fasth3.validate_profile_dependencies(official_args)

    load_started = time.perf_counter()
    generator = VideoGenerator.from_config(
        basic_fasth3.build_generator_config(official_args)
    )
    load_seconds = time.perf_counter() - load_started

    samples: list[dict[str, object]] = []
    try:
        warmup_path = outer.output_dir / "warmup.mp4"
        warmup_started = time.perf_counter()
        generator.generate(
            basic_fasth3.build_request(official_args, warmup_path, outer.warmup_seed)
        )
        warmup_seconds = time.perf_counter() - warmup_started

        for index in range(1, outer.repeats + 1):
            output = outer.output_dir / f"run-{index:02d}.mp4"
            sampler = MemorySampler(physical_gpus)
            with sampler:
                started = time.perf_counter()
                result = generator.generate(
                    basic_fasth3.build_request(official_args, output, outer.seed)
                )
                e2e_seconds = time.perf_counter() - started
            actual_output = basic_fasth3._actual_output_path(result, output)
            denoise_seconds = basic_fasth3._denoise_seconds(result)
            if denoise_seconds is None:
                raise RuntimeError("FastVideo result did not expose denoising time")
            generation_time = getattr(result, "generation_time", None)
            if generation_time is not None:
                generation_time = float(generation_time)
            validity = probe_mp4(
                actual_output, width=1344, height=768, frames=outer.num_frames
            )
            if not validity["valid"]:
                raise RuntimeError(f"invalid generated media: {validity['errors']}")
            samples.append(
                {
                    "index": index,
                    "output": str(actual_output),
                    "e2e_seconds": e2e_seconds,
                    "denoising_seconds": denoise_seconds,
                    "generation_time": generation_time,
                    "memory": sampler.summary(),
                    "validity": validity,
                }
            )
            del result
            gc.collect()
    finally:
        generator.shutdown()

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
        "framework": "FastVideo",
        "profile": "official-fasth3-dense-datafree-bf16-fa4",
        "commit": git_commit(outer.fastvideo_repo),
        "hardware": f"{outer.num_gpus}x NVIDIA H100 80GB HBM3",
        "physical_gpus": physical_gpus,
        "initial_memory_mib": initial_memory,
        "parallelism": {"tp_degree": 1, "sp_degree": outer.num_gpus},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": "MiniMaxAI/MiniMax-H3",
        "adapter": {
            "name": "FastH3 Dense/Data-Free",
            "sha256": file_sha256(outer.adapter_path),
            "strength": 1.0,
        },
        "workload": {
            "task": "t2va",
            "prompt": outer.prompt,
            "seed": outer.seed,
            "warmup_seed": outer.warmup_seed,
            "resolution": [1344, 768],
            "frames": outer.num_frames,
            "fps": 24,
            "duration_seconds": outer.duration_seconds,
            "sigma_points": 5,
            "actual_dit_forwards": 4,
        },
        "configuration": {
            "source": "examples/inference/basic/basic_fasth3_lora_preview.py",
            "profile": official_args.profile,
            "attention": "BF16 FA4",
            "h3_fusions": True,
            "regional_compile": False,
            "compile_vae": False,
            "parallel_vae": True,
            "replicated_dit": False,
            "fsdp_sharded_dit": True,
            "dit_offload": False,
            "lazy_module_load": True,
            "text_encoder_offload": True,
            "vae_offload": True,
            "boot_environment": environment,
            "source_files": {
                "example": str(Path(basic_fasth3.__file__).resolve()),
                "adapter_example": str(
                    Path(basic_fasth3_lora_preview.__file__).resolve()
                ),
            },
        },
        "measurement": {
            "scope": "prompt processing through synchronized MP4 close",
            "warmups": 1,
            "repeats": outer.repeats,
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
    write_report(outer.metrics_json, report)


if __name__ == "__main__":
    main()
