#!/usr/bin/env python3
"""Evaluate routing behavior for FAQ quick answers vs DOCX module guidance."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone

import requests


def _is_pass(case_mode: str, payload: dict) -> tuple[bool, str]:
    sources = payload.get("sources") or []
    answer = (payload.get("answer") or "").lower()

    if case_mode == "faq":
        if not sources:
            return False, "no sources"
        return (sources[0].get("source_type") == "faq"), f"top_source_type={sources[0].get('source_type')}"

    if case_mode == "docx":
        if not sources:
            return False, "no sources"
        return (sources[0].get("source_type") == "module_doc"), f"top_source_type={sources[0].get('source_type')}"

    if case_mode == "hybrid":
        has_docx = any((s.get("source_type") == "module_doc") for s in sources)
        has_hybrid_phrase = ("bung chi tiết" in answer) or ("muốn mình" in answer)
        return (has_docx and has_hybrid_phrase), f"has_docx={has_docx}, hybrid_phrase={has_hybrid_phrase}"

    return False, "unknown expected_mode"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DOCX routing behavior")
    parser.add_argument("--dataset", default="reports/eval/docx_routing_cases.json")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api/v1/ask")
    parser.add_argument("--out-json", default="reports/eval/docx_routing_report.json")
    parser.add_argument("--out-md", default="reports/eval/docx_routing_report.md")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep-ms", type=int, default=0)
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases") or []
    if not cases:
        raise RuntimeError("Dataset has no cases")

    details = []
    latencies = []
    pass_count = 0

    for idx, case in enumerate(cases, 1):
        query = case["query"]
        expected_mode = case["expected_mode"]
        payload = {
            "query": query,
            "session_id": f"docx-eval-{idx}",
            "channel": "api",
            "user_id": "eval",
        }

        started = time.time()
        resp = requests.post(args.api_url, json=payload, timeout=args.timeout)
        elapsed_ms = (time.time() - started) * 1000
        latencies.append(elapsed_ms)

        if not resp.ok:
            details.append(
                {
                    "case_id": case["case_id"],
                    "expected_mode": expected_mode,
                    "query": query,
                    "pass": False,
                    "reason": f"HTTP {resp.status_code}",
                    "latency_ms": round(elapsed_ms, 1),
                }
            )
            continue

        body = resp.json()
        ok, reason = _is_pass(expected_mode, body)
        if ok:
            pass_count += 1

        top_source = (body.get("sources") or [{}])[0]
        details.append(
            {
                "case_id": case["case_id"],
                "expected_mode": expected_mode,
                "query": query,
                "pass": ok,
                "reason": reason,
                "latency_ms": round(elapsed_ms, 1),
                "provider": body.get("provider"),
                "is_relevant": body.get("is_relevant"),
                "rewrite_attempts": body.get("rewrite_attempts"),
                "top_source_type": top_source.get("source_type"),
                "top_source_id": top_source.get("source_id") or top_source.get("issue_id"),
                "answer_preview": (body.get("answer") or "")[:220],
            }
        )

        if args.sleep_ms > 0 and idx < len(cases):
            time.sleep(args.sleep_ms / 1000)

    mode_stats = {}
    for mode in ["faq", "docx", "hybrid"]:
        subset = [d for d in details if d.get("expected_mode") == mode]
        if not subset:
            continue
        mode_stats[mode] = {
            "total": len(subset),
            "passed": sum(1 for d in subset if d.get("pass")),
            "pass_rate": round(sum(1 for d in subset if d.get("pass")) / len(subset), 4),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "cases": len(details),
        "passed": pass_count,
        "pass_rate": round(pass_count / max(1, len(details)), 4),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": round(sorted(latencies)[int(max(0, len(latencies) * 0.95 - 1))], 1) if latencies else 0.0,
        "mode_stats": mode_stats,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# DOCX Routing Evaluation Report\n\n")
        f.write(f"- Generated at: {summary['generated_at']}\n")
        f.write(f"- API: {summary['api_url']}\n")
        f.write(f"- Cases: {summary['cases']}\n")
        f.write(f"- Passed: {summary['passed']}\n")
        f.write(f"- Pass rate: {summary['pass_rate']:.2%}\n")
        f.write(f"- Latency p50: {summary['latency_p50_ms']} ms\n")
        f.write(f"- Latency p95: {summary['latency_p95_ms']} ms\n\n")

        f.write("## Mode Stats\n\n")
        for mode, stats in mode_stats.items():
            f.write(f"- {mode}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.2%})\n")

        f.write("\n## Failed Cases\n\n")
        failed = [d for d in details if not d.get("pass")]
        if not failed:
            f.write("No failed cases.\n")
        else:
            for item in failed:
                f.write(f"### {item['case_id']}\n")
                f.write(f"- Query: {item['query']}\n")
                f.write(f"- Expected: {item['expected_mode']}\n")
                f.write(f"- Reason: {item['reason']}\n")
                f.write(f"- Top source type: {item.get('top_source_type')}\n")
                f.write(f"- Top source id: {item.get('top_source_id')}\n")
                f.write(f"- Preview: {item.get('answer_preview', '')}\n\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report: {args.out_json}")
    print(f"Saved report: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
