#!/usr/bin/env python3
"""Simple concurrent load test for /api/v1/ask."""

from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests


@dataclass
class Result:
    ok: bool
    latency_ms: float
    error: str = ""


def _bucket_for(session_id: str) -> int:
    digest = hashlib.md5(session_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _build_payload(base_payload: dict, session_mode: str, worker_idx: int) -> dict:
    payload = dict(base_payload)
    if session_mode == "fixed":
        return payload
    if session_mode == "per-worker":
        payload["session_id"] = f"ltw-{worker_idx}"
        return payload
    if session_mode == "random":
        payload["session_id"] = f"ltr-{uuid.uuid4().hex[:12]}"
        return payload
    return payload


def worker_loop(stop_at: float, endpoint: str, base_payload: dict, timeout: float, out: list[Result], lock: threading.Lock, session_mode: str, worker_idx: int) -> None:
    while time.time() < stop_at:
        payload = _build_payload(base_payload, session_mode=session_mode, worker_idx=worker_idx)
        t0 = time.perf_counter()
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if resp.ok:
                out_item = Result(ok=True, latency_ms=latency_ms)
            else:
                out_item = Result(ok=False, latency_ms=latency_ms, error=f"HTTP {resp.status_code}")
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            out_item = Result(ok=False, latency_ms=latency_ms, error=str(exc))

        with lock:
            out.append(out_item)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent load test for /api/v1/ask")
    parser.add_argument("--endpoint", default="http://localhost:8000/api/v1/ask")
    parser.add_argument("--query", default="Test tải đồng thời")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--duration", type=int, default=60, help="seconds")
    parser.add_argument("--timeout", type=float, default=90.0, help="request timeout (seconds)")
    parser.add_argument("--channel", default="loadtest")
    parser.add_argument("--session-id", default="", help="Optional fixed session_id for deterministic rollout bucket")
    parser.add_argument(
        "--session-mode",
        choices=["fixed", "per-worker", "random"],
        default="fixed",
        help="How to assign session_id across requests/workers",
    )
    args = parser.parse_args()

    payload = {
        "query": args.query,
        "channel": args.channel,
        "user_id": "perf-user",
    }
    if args.session_id:
        payload["session_id"] = args.session_id

    results: list[Result] = []
    lock = threading.Lock()
    stop_at = time.time() + args.duration

    if args.session_id:
        sid_info = f"fixed:{args.session_id} bucket={_bucket_for(args.session_id)}"
    elif args.session_mode == "fixed":
        sid_info = "-"
    else:
        sid_info = args.session_mode

    print(
        "Start load test: "
        f"c={args.concurrency}, duration={args.duration}s, endpoint={args.endpoint}, "
        f"session={sid_info}"
    )
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(worker_loop, stop_at, args.endpoint, payload, args.timeout, results, lock, args.session_mode, idx)
            for idx in range(args.concurrency)
        ]
        for f in futures:
            f.result()
    elapsed = time.time() - start

    total = len(results)
    ok = sum(1 for r in results if r.ok)
    failed = total - ok
    latencies = sorted(r.latency_ms for r in results)

    rps = total / elapsed if elapsed > 0 else 0.0
    print("\n=== Summary ===")
    print(f"Total requests : {total}")
    print(f"Success        : {ok}")
    print(f"Failed         : {failed}")
    print(f"Error rate     : {(failed / total * 100.0) if total else 0.0:.2f}%")
    print(f"Throughput     : {rps:.2f} req/s ({rps*60:.0f} req/min)")

    if latencies:
        print(f"Latency avg    : {statistics.mean(latencies):.1f} ms")
        print(f"Latency p50    : {percentile(latencies, 50):.1f} ms")
        print(f"Latency p95    : {percentile(latencies, 95):.1f} ms")
        print(f"Latency p99    : {percentile(latencies, 99):.1f} ms")

    if failed:
        top_errors: dict[str, int] = {}
        for r in results:
            if r.ok:
                continue
            top_errors[r.error] = top_errors.get(r.error, 0) + 1
        print("Top errors:")
        for err, cnt in sorted(top_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {err}: {cnt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
