#!/usr/bin/env python3
"""Compare matched MiniMax-H3 MP4 video and audio without extracting files."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import BinaryIO

import numpy as np
from skimage.metrics import structural_similarity


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,width,height,nb_read_frames,channels,sample_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def video_metrics(
    reference: Path, candidate: Path, sample_stride: int
) -> dict[str, object]:
    metadata = probe(reference)
    video = next(
        stream for stream in metadata["streams"] if stream.get("codec_type") == "video"
    )
    width = int(video["width"])
    height = int(video["height"])
    frame_size = width * height * 3
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(reference),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    reference_process = subprocess.Popen(command, stdout=subprocess.PIPE)
    candidate_command = command.copy()
    candidate_command[4] = str(candidate)
    candidate_process = subprocess.Popen(candidate_command, stdout=subprocess.PIPE)
    if reference_process.stdout is None or candidate_process.stdout is None:
        raise RuntimeError("failed to open ffmpeg output")

    dot = reference_norm = candidate_norm = squared_error = 0.0
    samples = 0
    ssim_values: list[float] = []
    while True:
        reference_bytes = read_exact(reference_process.stdout, frame_size)
        candidate_bytes = read_exact(candidate_process.stdout, frame_size)
        if not reference_bytes and not candidate_bytes:
            break
        if len(reference_bytes) != frame_size or len(candidate_bytes) != frame_size:
            raise RuntimeError(
                "matched videos have different or incomplete frame counts"
            )
        reference_frame = np.frombuffer(reference_bytes, dtype=np.uint8).reshape(
            height, width, 3
        )
        candidate_frame = np.frombuffer(candidate_bytes, dtype=np.uint8).reshape(
            height, width, 3
        )
        reference_float = reference_frame.astype(np.float64)
        candidate_float = candidate_frame.astype(np.float64)
        dot += float(np.sum(reference_float * candidate_float))
        reference_norm += float(np.sum(reference_float * reference_float))
        candidate_norm += float(np.sum(candidate_float * candidate_float))
        squared_error += float(np.sum((reference_float - candidate_float) ** 2))
        if samples % sample_stride == 0:
            ssim_values.append(
                float(
                    structural_similarity(
                        reference_frame,
                        candidate_frame,
                        channel_axis=2,
                        data_range=255,
                    )
                )
            )
        samples += 1

    for process in (reference_process, candidate_process):
        if process.wait() != 0:
            raise RuntimeError("ffmpeg video decode failed")
    if samples == 0:
        raise RuntimeError("no video frames decoded")
    count = samples * frame_size
    mse = squared_error / count
    return {
        "frames": samples,
        "width": width,
        "height": height,
        "cosine_similarity": dot / math.sqrt(reference_norm * candidate_norm),
        "mse": mse,
        "psnr_db": None if mse == 0 else 10.0 * math.log10(255.0**2 / mse),
        "ssim_stride": sample_stride,
        "ssim_samples": len(ssim_values),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_minimum": float(np.min(ssim_values)),
    }


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, 2)


def magnitude_spectrogram(waveform: np.ndarray) -> np.ndarray:
    mono = waveform.mean(axis=1)
    window_size = 2048
    hop = 1024
    if mono.size < window_size:
        mono = np.pad(mono, (0, window_size - mono.size))
    count = 1 + (mono.size - window_size) // hop
    frames = np.lib.stride_tricks.sliding_window_view(mono, window_size)[::hop][:count]
    return np.abs(np.fft.rfft(frames * np.hanning(window_size), axis=1))


def audio_metrics(
    reference: Path, candidate: Path, sample_rate: int
) -> dict[str, object]:
    reference_audio = decode_audio(reference, sample_rate)
    candidate_audio = decode_audio(candidate, sample_rate)
    samples = min(len(reference_audio), len(candidate_audio))
    if samples == 0:
        raise RuntimeError("no audio samples decoded")
    reference_audio = reference_audio[:samples].astype(np.float64)
    candidate_audio = candidate_audio[:samples].astype(np.float64)
    error = reference_audio - candidate_audio
    reference_spectrum = magnitude_spectrogram(reference_audio)
    candidate_spectrum = magnitude_spectrogram(candidate_audio)
    epsilon = 1e-8
    return {
        "sample_rate": sample_rate,
        "channels": 2,
        "samples_per_channel": samples,
        "duration_seconds": samples / sample_rate,
        "cosine_similarity": float(
            np.vdot(reference_audio, candidate_audio)
            / (
                np.linalg.norm(reference_audio) * np.linalg.norm(candidate_audio)
                + epsilon
            )
        ),
        "mse": float(np.mean(error**2)),
        "spectral_convergence": float(
            np.linalg.norm(reference_spectrum - candidate_spectrum)
            / (np.linalg.norm(reference_spectrum) + epsilon)
        ),
        "log_spectral_distance_db": float(
            np.sqrt(
                np.mean(
                    (
                        20.0 * np.log10(reference_spectrum + epsilon)
                        - 20.0 * np.log10(candidate_spectrum + epsilon)
                    )
                    ** 2
                )
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssim-stride", type=int, default=8)
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    args = parser.parse_args()
    if args.ssim_stride < 1:
        parser.error("--ssim-stride must be positive")

    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "video": video_metrics(args.reference, args.candidate, args.ssim_stride),
        "audio": audio_metrics(args.reference, args.candidate, args.audio_sample_rate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
