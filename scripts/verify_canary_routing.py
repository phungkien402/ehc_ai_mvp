#!/usr/bin/env python3
"""Verify deterministic canary routing and provider assignment on /api/v1/ask."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass

import requests


@dataclass
class ProbeResult:
    session_id: str
    bucket: int
    expected_provider: str
    actual_provider: str
    status_code: int
    ok: bool
    error: str = ""


def bucket_for(session_id: str) -> int:
    digest = hashlib.md5(session_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def find_session(target_lt: bool, threshold: int, start: int = 1, end: int = 20000) -> tuple[str, int]:
    for i in range(start, end + 1):
        sid = f"s{i}"
        b = bucket_for(sid)
        if target_lt and b < threshold:
            return sid, b
        if not target_lt and b >= threshold:
            return sid, b
    raise RuntimeError("No suitable session_id found in search range")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify canary routing behavior")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/v1/ask")
    parser.add_argument("--rollout-percent", type=int, default=10)
    parser.add_argument("--base-provider", default="ollama")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--query", default="xin chao")
    args = parser.parse_args()

    if args.runs < 1:
        print("runs must be >= 1", file=sys.stderr)
        return 2

    rollout_percent = max(0, min(args.rollout_percent, 100))
    base_provider = (args.base_provider or "ollama").strip().lower()

    if rollout_percent == 0:
        canary_sid, canary_bucket = "s3", bucket_for("s3")
        control_sid, control_bucket = "s1", bucket_for("s1")
        cases = [
            (canary_sid, base_provider),
            (control_sid, base_provider),
        ]
    elif rollout_percent == 100:
        canary_sid, canary_bucket = "s3", bucket_for("s3")
        control_sid, control_bucket = "s1", bucket_for("s1")
        cases = [
            (canary_sid, "vllm"),
            (control_sid, "vllm"),
        ]
    else:
        canary_sid, canary_bucket = find_session(True, rollout_percent)
        control_sid, control_bucket = find_session(False, rollout_percent)
        cases = [
            (canary_sid, "vllm"),
            (control_sid, base_provider),
        ]

    print("Selected sessions:")
    print(f"  canary:  {canary_sid} (bucket={canary_bucket})")
    print(f"  control: {control_sid} (bucket={control_bucket})")
    print(f"  rollout_percent={rollout_percent} base_provider={base_provider}")

    all_results: list[ProbeResult] = []

    for sid, expected in cases:
        for _ in range(args.runs):
            payload = {
                "query": args.query,
                "session_id": sid,
                "channel": "api",
                "user_id": "canary-checker",
            }
            try:
                resp = requests.post(args.endpoint, json=payload, timeout=args.timeout)
            except Exception as exc:
                all_results.append(
                    ProbeResult(
                        session_id=sid,
                        bucket=bucket_for(sid),
                        expected_provider=expected,
                        actual_provider="",
                        status_code=0,
                        ok=False,
                        error=str(exc),
                    )
                )
                continue

            actual = ""
            err = ""
            if resp.ok:
                try:
                    body = resp.json()
                    actual = str(body.get("provider", ""))
                except json.JSONDecodeError as exc:
                    err = f"invalid json: {exc}"
            else:
                err = f"HTTP {resp.status_code}"

            ok = resp.ok and actual == expected
            all_results.append(
                ProbeResult(
                    session_id=sid,
                    bucket=bucket_for(sid),
                    expected_provider=expected,
                    actual_provider=actual,
                    status_code=resp.status_code,
                    ok=ok,
                    error=err,
                )
            )

    failures = [r for r in all_results if not r.ok]
    passed = len(all_results) - len(failures)

    print("\nSummary:")
    print(f"  total={len(all_results)} passed={passed} failed={len(failures)}")

    if failures:
        print("\nFailures:")
        for f in failures[:20]:
            print(
                f"  sid={f.session_id} bucket={f.bucket} expected={f.expected_provider} "
                f"actual={f.actual_provider or '-'} status={f.status_code} err={f.error or '-'}"
            )
        return 1

    print("  deterministic routing verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
