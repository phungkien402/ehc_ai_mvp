#!/usr/bin/env python3
"""Benchmark vLLM (OpenAI-compatible) and capture GPU hardware telemetry.

This script is designed for 1-2 hour hardware validation runs and writes
both JSON and Markdown reports with latency and temperature/power/VRAM traces.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_PROMPT = (
    "Viết ngắn gọn 5 câu mô tả cách đánh giá hiệu năng GPU cho hệ thống AI, "
    "bao gồm latency, throughput, VRAM, nhiệt độ và công suất."
)


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class RunResult:
    run: int
    started_at: str
    elapsed_sec: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    tokens_per_sec: float | None
    temp_max_c: float | None
    power_max_w: float | None
    memory_max_mib: float | None
    response_preview: str


@dataclass
class SensorSample:
    timestamp: str
    gpu_index: int
    gpu_name: str
    temp_c: float | None
    power_w: float | None
    power_limit_w: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    util_gpu_pct: float | None


def _pct(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[int(rank)]
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _format_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _hostname() -> str:
    try:
        return subprocess.check_output(["hostname", "-f"], text=True).strip()
    except Exception:
        try:
            return subprocess.check_output(["hostname"], text=True).strip()
        except Exception:
            return "unknown"


def _run_nvitop_snapshot() -> str:
    try:
        result = subprocess.run(
            ["nvitop", "--once", "--ascii", "--compute"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        output = result.stdout.strip()
        if result.stderr and result.stderr.strip():
            output = f"{output}\n\n[stderr]\n{result.stderr.strip()}".strip()
        return output or "(no output)"
    except FileNotFoundError:
        return "nvitop not found in PATH"
    except subprocess.CalledProcessError as exc:
        parts = [part.strip() for part in [exc.stdout or "", exc.stderr or ""] if part.strip()]
        return "\n".join(parts) if parts else f"nvitop failed: {exc}"
    except subprocess.TimeoutExpired:
        return "nvitop snapshot timed out"


def _sample_nvidia_smi() -> list[SensorSample]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    timestamp = datetime.now().isoformat(timespec="seconds")
    samples: list[SensorSample] = []
    for raw_line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 8:
            continue
        try:
            samples.append(
                SensorSample(
                    timestamp=timestamp,
                    gpu_index=int(parts[0]),
                    gpu_name=parts[1],
                    temp_c=float(parts[2]) if parts[2] else None,
                    power_w=float(parts[3]) if parts[3] else None,
                    power_limit_w=float(parts[4]) if parts[4] else None,
                    memory_used_mib=float(parts[5]) if parts[5] else None,
                    memory_total_mib=float(parts[6]) if parts[6] else None,
                    util_gpu_pct=float(parts[7]) if parts[7] else None,
                )
            )
        except ValueError:
            continue
    return samples


def _http_json(url: str, payload: dict | None = None, headers: dict | None = None, timeout: int = 120) -> dict:
    encoded = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=encoded, headers=request_headers, method="POST" if payload else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _auth_headers(api_key: str) -> dict:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _check_vllm_models(base_url: str, api_key: str) -> None:
    _http_json(f"{base_url.rstrip('/')}/v1/models", headers=_auth_headers(api_key), timeout=30)


def _vllm_chat_completion(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    return _http_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        payload=payload,
        headers=_auth_headers(api_key),
        timeout=180,
    )


def _summarize(runs: list[RunResult]) -> dict:
    elapsed_values = [item.elapsed_sec for item in runs]
    tps_values = [item.tokens_per_sec for item in runs if item.tokens_per_sec is not None]
    temp_values = [item.temp_max_c for item in runs if item.temp_max_c is not None]
    power_values = [item.power_max_w for item in runs if item.power_max_w is not None]
    mem_values = [item.memory_max_mib for item in runs if item.memory_max_mib is not None]
    return {
        "samples": len(runs),
        "elapsed_avg_sec": statistics.mean(elapsed_values),
        "elapsed_min_sec": min(elapsed_values),
        "elapsed_max_sec": max(elapsed_values),
        "elapsed_p50_sec": _pct(sorted(elapsed_values), 50),
        "elapsed_p95_sec": _pct(sorted(elapsed_values), 95),
        "elapsed_stdev_sec": statistics.stdev(elapsed_values) if len(elapsed_values) > 1 else 0.0,
        "tokens_per_sec_avg": statistics.mean(tps_values) if tps_values else None,
        "temp_max_c": max(temp_values) if temp_values else None,
        "temp_avg_c": statistics.mean(temp_values) if temp_values else None,
        "power_max_w": max(power_values) if power_values else None,
        "memory_max_mib": max(mem_values) if mem_values else None,
    }


def _build_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# vLLM GPU Benchmark Report",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Host: {report['host']}",
        f"- Base URL: {report['base_url']}",
        f"- Model: {report['model']}",
        f"- Mode: {report['mode']}",
        f"- Duration minutes: {report.get('duration_minutes', 0)}",
        f"- Samples: {summary['samples']}",
        f"- Warmup runs: {report['warmup_runs']}",
        f"- Max tokens: {report['max_tokens']}",
        "",
        "## Summary",
        "",
        f"- Avg latency: {_format_seconds(summary['elapsed_avg_sec'])} s",
        f"- P95 latency: {_format_seconds(summary['elapsed_p95_sec'])} s",
        f"- Avg completion throughput: {_format_number(summary.get('tokens_per_sec_avg'), 2)} tok/s",
        f"- Max temperature: {_format_number(summary.get('temp_max_c'))} C",
        f"- Max power: {_format_number(summary.get('power_max_w'))} W",
        f"- Max VRAM: {_format_number(summary.get('memory_max_mib'))} MiB",
        "",
        "## nvitop Snapshot",
        "",
        "```",
        report["nvitop_snapshot"],
        "```",
        "",
        "## Runs",
        "",
        "| Run | Started At | Elapsed (s) | Prompt Tok | Completion Tok | Tok/s | Temp Max (C) | Power Max (W) | VRAM Max (MiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in report["runs"]:
        lines.append(
            "| {run} | {started} | {elapsed} | {pt} | {ct} | {tps} | {temp} | {power} | {mem} |".format(
                run=item["run"],
                started=item["started_at"],
                elapsed=_format_seconds(item["elapsed_sec"]),
                pt=item["prompt_tokens"] if item["prompt_tokens"] is not None else "-",
                ct=item["completion_tokens"] if item["completion_tokens"] is not None else "-",
                tps=_format_number(item["tokens_per_sec"], 2),
                temp=_format_number(item["temp_max_c"]),
                power=_format_number(item["power_max_w"]),
                mem=_format_number(item["memory_max_mib"]),
            )
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This benchmark targets vLLM OpenAI-compatible endpoints.",
        "- Sensor samples are taken from nvidia-smi after each request loop.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")

    parser = argparse.ArgumentParser(description="Benchmark vLLM and record GPU telemetry")
    parser.add_argument("--base-url", default=os.getenv("VLLM_LLM_URL", os.getenv("VLLM_BASE_URL", "http://localhost:8080")))
    parser.add_argument("--model", default=os.getenv("VLLM_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--duration-minutes", type=float, default=0)
    parser.add_argument("--sample-every-seconds", type=float, default=0)
    parser.add_argument("--out-dir", default="reports/gpu_benchmarks")
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit("--samples must be > 0")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be >= 0")
    if args.duration_minutes < 0:
        raise SystemExit("--duration-minutes must be >= 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"Checking vLLM endpoint: {args.base_url}")
    try:
        _check_vllm_models(args.base_url, args.api_key)
    except Exception as exc:
        raise SystemExit(f"vLLM endpoint not ready: {exc}") from exc

    print(f"Benchmarking model={args.model} | mode={'duration' if args.duration_minutes > 0 else 'sample-count'}")
    for idx in range(args.warmup_runs):
        print(f"Warmup {idx + 1}/{args.warmup_runs}...")
        _vllm_chat_completion(args.base_url, args.api_key, args.model, args.prompt, args.max_tokens)

    runs: list[RunResult] = []
    sensor_samples: list[SensorSample] = []
    nvitop_snapshot = _run_nvitop_snapshot()

    if args.duration_minutes > 0:
        deadline = time.monotonic() + args.duration_minutes * 60.0
        run_idx = 0
        while time.monotonic() < deadline:
            run_idx += 1
            started_at = datetime.now().isoformat(timespec="seconds")
            start_perf = time.perf_counter()
            data = _vllm_chat_completion(args.base_url, args.api_key, args.model, args.prompt, args.max_tokens)
            elapsed = time.perf_counter() - start_perf
            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            total_tokens = usage.get("total_tokens")
            tps = (float(completion_tokens) / elapsed) if completion_tokens and elapsed > 0 else None
            sensors = _sample_nvidia_smi()
            sensor_samples.extend(sensors)
            temp_values = [s.temp_c for s in sensors if s.temp_c is not None]
            power_values = [s.power_w for s in sensors if s.power_w is not None]
            mem_values = [s.memory_used_mib for s in sensors if s.memory_used_mib is not None]
            content = ""
            choices = data.get("choices") or []
            if choices:
                content = ((choices[0].get("message") or {}).get("content") or "")[:250]
            runs.append(
                RunResult(
                    run=run_idx,
                    started_at=started_at,
                    elapsed_sec=elapsed,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    tokens_per_sec=tps,
                    temp_max_c=max(temp_values) if temp_values else None,
                    power_max_w=max(power_values) if power_values else None,
                    memory_max_mib=max(mem_values) if mem_values else None,
                    response_preview=content,
                )
            )
            print(f"Run {run_idx}: elapsed={elapsed:.2f}s, completion_tps={_format_number(tps, 2)} tok/s")
            if args.sample_every_seconds > 0:
                time.sleep(args.sample_every_seconds)
    else:
        for idx in range(args.samples):
            started_at = datetime.now().isoformat(timespec="seconds")
            start_perf = time.perf_counter()
            data = _vllm_chat_completion(args.base_url, args.api_key, args.model, args.prompt, args.max_tokens)
            elapsed = time.perf_counter() - start_perf
            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            total_tokens = usage.get("total_tokens")
            tps = (float(completion_tokens) / elapsed) if completion_tokens and elapsed > 0 else None
            sensors = _sample_nvidia_smi()
            sensor_samples.extend(sensors)
            temp_values = [s.temp_c for s in sensors if s.temp_c is not None]
            power_values = [s.power_w for s in sensors if s.power_w is not None]
            mem_values = [s.memory_used_mib for s in sensors if s.memory_used_mib is not None]
            content = ""
            choices = data.get("choices") or []
            if choices:
                content = ((choices[0].get("message") or {}).get("content") or "")[:250]
            runs.append(
                RunResult(
                    run=idx + 1,
                    started_at=started_at,
                    elapsed_sec=elapsed,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    tokens_per_sec=tps,
                    temp_max_c=max(temp_values) if temp_values else None,
                    power_max_w=max(power_values) if power_values else None,
                    memory_max_mib=max(mem_values) if mem_values else None,
                    response_preview=content,
                )
            )
            print(f"Run {idx + 1}/{args.samples}: elapsed={elapsed:.2f}s, completion_tps={_format_number(tps, 2)} tok/s")

    summary = _summarize(runs)
    mode = "duration" if args.duration_minutes > 0 else "sample-count"
    report = {
        "timestamp": timestamp,
        "host": _hostname(),
        "base_url": args.base_url,
        "model": args.model,
        "mode": mode,
        "duration_minutes": args.duration_minutes,
        "sample_every_seconds": args.sample_every_seconds,
        "warmup_runs": args.warmup_runs,
        "samples": len(runs),
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "nvitop_snapshot": nvitop_snapshot,
        "sensor_samples": [asdict(item) for item in sensor_samples],
        "summary": summary,
        "runs": [asdict(item) for item in runs],
    }

    json_path = out_dir / f"vllm_gpu_benchmark_{timestamp}.json"
    md_path = out_dir / f"vllm_gpu_benchmark_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")

    print("\nSaved report:")
    print(f"- {json_path}")
    print(f"- {md_path}")
    print("\nSummary:")
    print(f"- Avg latency: {_format_seconds(summary['elapsed_avg_sec'])} s")
    print(f"- P95 latency: {_format_seconds(summary['elapsed_p95_sec'])} s")
    print(f"- Avg completion throughput: {_format_number(summary.get('tokens_per_sec_avg'), 2)} tok/s")
    print(f"- Max temperature: {_format_number(summary.get('temp_max_c'))} C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())