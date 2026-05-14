#!/usr/bin/env python3
"""Simple load runner for WhisperIO profile benchmarking."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import mimetypes
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def post_job(api_base: str, file_path: Path) -> str:
    boundary = "----whisperio-benchmark-boundary"
    filename = file_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_data = file_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        f"{api_base}/api/jobs",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["job_id"])


def get_job(api_base: str, job_id: str) -> dict:
    request = urllib.request.Request(f"{api_base}/api/jobs/{job_id}", method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_done(api_base: str, job_id: str, poll_sec: float) -> dict:
    while True:
        payload = get_job(api_base, job_id)
        status = payload.get("status")
        if status in {"done", "failed"}:
            return payload
        time.sleep(poll_sec)


def probe_duration_sec(file_path: Path, timeout_sec: int) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(5, timeout_sec),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
        return max(0.0, float(payload.get("format", {}).get("duration", 0.0)))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def run_case(
    api_base: str,
    file_path: Path,
    poll_sec: float,
    profile_name: str,
    probe_timeout_sec: int,
    run_index: int,
) -> dict:
    audio_duration_sec = probe_duration_sec(file_path, timeout_sec=probe_timeout_sec)
    started = time.perf_counter()
    job_id = post_job(api_base, file_path)
    payload = wait_done(api_base, job_id, poll_sec=poll_sec)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    processing_duration_ms = payload.get("processing_duration_ms")
    quality_flags = payload.get("quality_flags")
    rescue_applied_count = None
    vad_window_count = None
    rescue_window_count = None
    if isinstance(quality_flags, str):
        try:
            parsed_flags = json.loads(quality_flags)
        except json.JSONDecodeError:
            parsed_flags = None
        if isinstance(parsed_flags, dict):
            rescue_applied_count = parsed_flags.get("rescue_applied_count")
            vad_window_count = parsed_flags.get("vad_window_count")
            rescue_window_count = parsed_flags.get("rescue_window_count")
    processing_rtf = None
    effective_speed_x = None
    if audio_duration_sec and processing_duration_ms:
        processing_sec = float(processing_duration_ms) / 1000.0
        if processing_sec > 0:
            processing_rtf = round(processing_sec / audio_duration_sec, 6)
            effective_speed_x = round(audio_duration_sec / processing_sec, 6)
    return {
        "job_id": job_id,
        "file": str(file_path),
        "run_index": run_index,
        "profile": profile_name,
        "status": payload.get("status"),
        "audio_duration_sec": audio_duration_sec,
        "processing_duration_ms": processing_duration_ms,
        "transcribe_duration_ms": payload.get("transcribe_duration_ms"),
        "processing_rtf": processing_rtf,
        "effective_speed_x": effective_speed_x,
        "elapsed_wall_ms": elapsed_ms,
        "quality_flags": quality_flags,
        "vad_window_count": vad_window_count,
        "rescue_window_count": rescue_window_count,
        "rescue_applied_count": rescue_applied_count,
        "error": payload.get("error"),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * p))))
    return ordered[rank]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WhisperIO profiles via API.")
    parser.add_argument(
        "--api-base",
        default=os.getenv("WHISPER_BENCHMARK_API", "http://localhost:8000"),
        help="WhisperIO API base URL",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Audio files for one benchmark run",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Parallel files to execute",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=2.0,
        help="Polling interval while waiting job completion",
    )
    parser.add_argument(
        "--out",
        default="benchmark-results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--profile-name",
        default=os.getenv("TRANSCRIBE_PROFILE", os.getenv("WHISPER_BENCHMARK_PROFILE", "default")),
        help="Label for backend whisper profile used in this run",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each file multiple times",
    )
    parser.add_argument(
        "--probe-timeout-sec",
        type=int,
        default=30,
        help="ffprobe timeout while extracting media duration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = [Path(item) for item in args.files]
    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

    execution_cases: list[tuple[Path, int]] = []
    for file_path in files:
        for run_index in range(1, max(1, args.repeat) + 1):
            execution_cases.append((file_path, run_index))

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [
            pool.submit(
                run_case,
                args.api_base.rstrip("/"),
                file_path,
                args.poll_sec,
                args.profile_name,
                args.probe_timeout_sec,
                run_index,
            )
            for file_path, run_index in execution_cases
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{result['status']}] profile={result['profile']} run={result['run_index']} {result['file']} "
                    f"processing_ms={result['processing_duration_ms']} "
                    f"transcribe_ms={result['transcribe_duration_ms']} "
                    f"rtf={result['processing_rtf']}"
                )
            except urllib.error.HTTPError as exc:
                print(f"[http-error] code={exc.code} message={exc.reason}")
            except Exception as exc:  # noqa: BLE001
                print(f"[error] {exc}")

    processing_values = [
        int(item["processing_duration_ms"])
        for item in results
        if item.get("processing_duration_ms") is not None
    ]
    rtf_values = [item["processing_rtf"] for item in results if item.get("processing_rtf") is not None]
    summary = {
        "profile_name": args.profile_name,
        "files_count": len(files),
        "repeat_per_file": max(1, args.repeat),
        "total_jobs": len(results),
        "done_jobs": sum(1 for item in results if item["status"] == "done"),
        "failed_jobs": sum(1 for item in results if item["status"] == "failed"),
        "p50_processing_ms": percentile(processing_values, 0.50),
        "p95_processing_ms": percentile(processing_values, 0.95),
        "p50_processing_rtf": percentile(rtf_values, 0.50) if rtf_values else 0,
        "p95_processing_rtf": percentile(rtf_values, 0.95) if rtf_values else 0,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved benchmark report to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
