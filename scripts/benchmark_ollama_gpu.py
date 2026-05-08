#!/usr/bin/env python3
"""Benchmark local Ollama GPU inference and capture temperature/power/VRAM over time.

Two modes are supported:
- fixed sample mode: run N requests and summarize latency / token throughput
- duration mode: run continuously for 1-2 hours and record GPU sensors over time
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_PROMPT = (
    "Viết một đoạn ngắn 5 câu về mục tiêu của benchmark GPU, "
    "tập trung vào throughput, latency, VRAM và nhiệt độ."
)


@dataclass
class RunResult:
    run: int
    started_at: str
    elapsed_sec: float
    total_duration_sec: float | None
    load_duration_sec: float | None
    prompt_eval_count: int | None
    eval_count: int | None
    eval_rate_tps: float | None
    prompt_rate_tps: float | None
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


def _ns_to_sec(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return None


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


def _format_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


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


def _ollama_generate(base_url: str, model: str, prompt: str, num_predict: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _build_markdown(report: dict) -> str:
    summary = report["summary"]
    duration_mode = "duration" if report.get("duration_minutes", 0) else "sample-count"
    lines = [
        "# GPU Benchmark Report",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Host: {report['host']}",
        f"- Model: {report['model']}",
        f"- Base URL: {report['base_url']}",
        f"- Mode: {duration_mode}",
        f"- Samples: {summary['samples']}",
        f"- Duration minutes: {report.get('duration_minutes', 0)}",
        f"- Warmup runs: {report['warmup_runs']}",
        f"- Prompt tokens: {report['prompt_tokens']}",
        f"- Requested output tokens: {report['num_predict']}",
        "",
        "## Summary",
        "",
        f"- Avg latency: {_format_seconds(summary['elapsed_avg_sec'])} s",
        f"- Min latency: {_format_seconds(summary['elapsed_min_sec'])} s",
        f"- Max latency: {_format_seconds(summary['elapsed_max_sec'])} s",
        f"- P50 latency: {_format_seconds(summary['elapsed_p50_sec'])} s",
        f"- P95 latency: {_format_seconds(summary['elapsed_p95_sec'])} s",
        f"- Stddev: {_format_seconds(summary['elapsed_stdev_sec'])} s",
        f"- Avg eval rate: {_format_rate(summary['eval_rate_avg_tps'])} tok/s",
        f"- Avg prompt eval rate: {_format_rate(summary['prompt_rate_avg_tps'])} tok/s",
        f"- Max temp observed: {_format_number(summary.get('temp_max_c'))} °C",
        f"- Avg temp observed: {_format_number(summary.get('temp_avg_c'))} °C",
        f"- Max power observed: {_format_number(summary.get('power_max_w'))} W",
        f"- Max VRAM used: {_format_number(summary.get('memory_max_mib'))} MiB",
        "",
        "## nvitop Snapshot",
        "",
        "```",
        report["nvitop_snapshot"],
        "```",
        "",
        "## Runs",
        "",
        "| Run | Started At | Elapsed (s) | Total (s) | Load (s) | Temp Max (C) | Power Max (W) | VRAM Max (MiB) | Eval rate (tok/s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in report["runs"]:
        lines.append(
            "| {run} | {started} | {elapsed} | {total} | {load} | {temp} | {power} | {mem} | {rate} |".format(
                run=item["run"],
                started=item.get("started_at", "-"),
                elapsed=_format_seconds(item["elapsed_sec"]),
                total=_format_seconds(item["total_duration_sec"]),
                load=_format_seconds(item["load_duration_sec"]),
                temp=_format_number(item.get("temp_max_c")),
                power=_format_number(item.get("power_max_w")),
                mem=_format_number(item.get("memory_max_mib")),
                rate=_format_rate(item["eval_rate_tps"]),
            )
        )

    if report.get("sensor_samples"):
        lines.extend([
            "",
            "## Sensor Samples",
            "",
            "| Time | GPU | Temp (C) | Power (W) | VRAM Used (MiB) | Util (%) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for item in report["sensor_samples"]:
            lines.append(
                "| {time} | {gpu} | {temp} | {power} | {mem} | {util} |".format(
                    time=item["timestamp"],
                    gpu=item["gpu_index"],
                    temp=_format_number(item["temp_c"]),
                    power=_format_number(item["power_w"]),
                    mem=_format_number(item["memory_used_mib"]),
                    util=_format_number(item["util_gpu_pct"]),
                )
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This report measures Ollama inference latency and GPU residency, not gaming or graphics benchmarks.",
        "- For repeatability, use the same prompt, model, and token budget when comparing runs.",
        "- In duration mode, the script keeps issuing requests for the chosen time window and logs GPU sensors over time.",
        "",
    ])
    return "\n".join(lines)


def _summarize_runs(runs: list[RunResult]) -> dict:
    elapsed_values = [item.elapsed_sec for item in runs]
    eval_rate_values = [item.eval_rate_tps for item in runs if item.eval_rate_tps is not None]
    prompt_rate_values = [item.prompt_rate_tps for item in runs if item.prompt_rate_tps is not None]
    temp_max_values = [item.temp_max_c for item in runs if getattr(item, "temp_max_c", None) is not None]
    temp_avg_values = [item.temp_avg_c for item in runs if getattr(item, "temp_avg_c", None) is not None]
    power_max_values = [item.power_max_w for item in runs if getattr(item, "power_max_w", None) is not None]
    memory_max_values = [item.memory_max_mib for item in runs if getattr(item, "memory_max_mib", None) is not None]

    return {
        "samples": len(runs),
        "elapsed_avg_sec": statistics.mean(elapsed_values),
        "elapsed_min_sec": min(elapsed_values),
        "elapsed_max_sec": max(elapsed_values),
        "elapsed_p50_sec": _pct(sorted(elapsed_values), 50),
        "elapsed_p95_sec": _pct(sorted(elapsed_values), 95),
        "elapsed_stdev_sec": statistics.stdev(elapsed_values) if len(elapsed_values) > 1 else 0.0,
        "eval_rate_avg_tps": statistics.mean(eval_rate_values) if eval_rate_values else None,
        "prompt_rate_avg_tps": statistics.mean(prompt_rate_values) if prompt_rate_values else None,
        "temp_max_c": max(temp_max_values) if temp_max_values else None,
        "temp_avg_c": statistics.mean(temp_avg_values) if temp_avg_values else None,
        "power_max_w": max(power_max_values) if power_max_values else None,
        "memory_max_mib": max(memory_max_values) if memory_max_values else None,
    }


def _record_run(
    run_idx: int,
    started_at: str,
    start_perf: float,
    data: dict,
) -> RunResult:
    elapsed = time.perf_counter() - start_perf
    prompt_eval_count = data.get("prompt_eval_count")
    eval_count = data.get("eval_count")
    total_duration_sec = _ns_to_sec(data.get("total_duration"))
    load_duration_sec = _ns_to_sec(data.get("load_duration"))
    eval_duration_sec = _ns_to_sec(data.get("eval_duration"))
    prompt_eval_duration_sec = _ns_to_sec(data.get("prompt_eval_duration"))

    eval_rate_tps = None
    if eval_count and eval_duration_sec and eval_duration_sec > 0:
        eval_rate_tps = eval_count / eval_duration_sec

    prompt_rate_tps = None
    if prompt_eval_count and prompt_eval_duration_sec and prompt_eval_duration_sec > 0:
        prompt_rate_tps = prompt_eval_count / prompt_eval_duration_sec

    return RunResult(
        run=run_idx,
        started_at=started_at,
        elapsed_sec=elapsed,
        total_duration_sec=total_duration_sec,
        load_duration_sec=load_duration_sec,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        eval_rate_tps=eval_rate_tps,
        prompt_rate_tps=prompt_rate_tps,
        response_preview=(data.get("response", "") or "")[:300],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local Ollama GPU inference and capture a nvitop snapshot")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model to benchmark")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the model")
    parser.add_argument("--samples", type=int, default=3, help="Number of benchmark samples in sample-count mode")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup requests excluded from stats")
    parser.add_argument("--num-predict", type=int, default=128, help="Requested output token budget")
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=0,
        help="Run continuously for this many minutes instead of a fixed sample count",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=0,
        help="Optional pause between requests in duration mode",
    )
    parser.add_argument(
        "--sensor-every-seconds",
        type=float,
        default=15,
        help="Recommended target interval for sensor logging in duration mode",
    )
    parser.add_argument("--out-dir", default="reports/gpu_benchmarks", help="Directory for JSON/MD output")
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit("--samples must be > 0")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be >= 0")
    if args.duration_minutes < 0:
        raise SystemExit("--duration-minutes must be >= 0")
    if args.sample_every_seconds < 0:
        raise SystemExit("--sample-every-seconds must be >= 0")
    if args.sensor_every_seconds < 0:
        raise SystemExit("--sensor-every-seconds must be >= 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    mode_label = f"duration={args.duration_minutes}m" if args.duration_minutes else f"samples={args.samples}"
    print(f"Benchmarking model={args.model} base_url={args.base_url}")
    print(
        f"Mode: {mode_label}, warmup runs: {args.warmup_runs}, num_predict: {args.num_predict}, "
        f"sensor interval target: {args.sensor_every_seconds}s"
    )

    for idx in range(args.warmup_runs):
        print(f"Warmup {idx + 1}/{args.warmup_runs}...")
        _ollama_generate(args.base_url, args.model, args.prompt, args.num_predict)

    nvitop_snapshot = _run_nvitop_snapshot()

    runs: list[RunResult] = []
    sensor_samples: list[SensorSample] = []

    if args.duration_minutes > 0:
        deadline = time.monotonic() + (args.duration_minutes * 60.0)
        next_sensor_at = time.monotonic()
        run_idx = 0
        while time.monotonic() < deadline:
            run_idx += 1
            started_at = datetime.now().isoformat(timespec="seconds")
            start_perf = time.perf_counter()
            try:
                data = _ollama_generate(args.base_url, args.model, args.prompt, args.num_predict)
            except urllib.error.URLError as exc:
                raise SystemExit(f"Failed to call Ollama: {exc}") from exc

            run = _record_run(run_idx, started_at, start_perf, data)
            runs.append(run)

            current_sensors = _sample_nvidia_smi()
            sensor_samples.extend(current_sensors)
            temp_values = [s.temp_c for s in current_sensors if s.temp_c is not None]
            temp_text = _format_number(max(temp_values) if temp_values else None)
            print(
                f"Run {run_idx}: elapsed={run.elapsed_sec:.2f}s, temp={temp_text}C, "
                f"eval_rate={_format_rate(run.eval_rate_tps)} tok/s"
            )

            if args.sample_every_seconds > 0:
                time.sleep(args.sample_every_seconds)
            if time.monotonic() >= next_sensor_at:
                next_sensor_at = time.monotonic() + args.sensor_every_seconds
    else:
        for idx in range(args.samples):
            started_at = datetime.now().isoformat(timespec="seconds")
            start_perf = time.perf_counter()
            try:
                data = _ollama_generate(args.base_url, args.model, args.prompt, args.num_predict)
            except urllib.error.URLError as exc:
                raise SystemExit(f"Failed to call Ollama: {exc}") from exc
            run = _record_run(idx + 1, started_at, start_perf, data)
            runs.append(run)
            print(
                f"Run {idx + 1}/{args.samples}: elapsed={run.elapsed_sec:.2f}s, "
                f"eval_rate={_format_rate(run.eval_rate_tps)} tok/s, "
                f"prompt_tokens={run.prompt_eval_count}, eval_tokens={run.eval_count}"
            )

    summary = _summarize_runs(runs)

    report = {
        "timestamp": timestamp,
        "host": _hostname(),
        "base_url": args.base_url,
        "model": args.model,
        "prompt": args.prompt,
        "prompt_tokens": len(args.prompt.split()),
        "warmup_runs": args.warmup_runs,
        "samples": args.samples if args.duration_minutes == 0 else len(runs),
        "duration_minutes": args.duration_minutes,
        "sample_every_seconds": args.sample_every_seconds,
        "sensor_every_seconds": args.sensor_every_seconds,
        "num_predict": args.num_predict,
        "mode": "duration" if args.duration_minutes > 0 else "sample-count",
        "nvitop_snapshot": nvitop_snapshot,
        "sensor_samples": [asdict(item) for item in sensor_samples],
        "summary": summary,
        "runs": [asdict(item) for item in runs],
    }

    json_path = out_dir / f"ollama_gpu_benchmark_{timestamp}.json"
    md_path = out_dir / f"ollama_gpu_benchmark_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")

    print("\nSaved report:")
    print(f"- {json_path}")
    print(f"- {md_path}")
    print("\nSummary:")
    print(f"- Avg latency: {summary['elapsed_avg_sec']:.2f}s")
    print(f"- P95 latency: {summary['elapsed_p95_sec']:.2f}s")
    if summary["eval_rate_avg_tps"] is not None:
        print(f"- Avg eval rate: {summary['eval_rate_avg_tps']:.2f} tok/s")
    if summary["temp_max_c"] is not None:
        print(f"- Max temp observed: {summary['temp_max_c']:.1f} C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())