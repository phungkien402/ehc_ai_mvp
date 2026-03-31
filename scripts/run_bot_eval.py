#!/usr/bin/env python3
"""Run bot evaluation against ticket-derived test set.

Metrics per case:
- source_hit: predicted top source issue_id matches expected_issue_id
- semantic_similarity: cosine similarity between answer and reference (embedding)
- keyword_coverage: overlap of important terms from reference found in answer
- readability_ok: answer length and no raw-tool leakage patterns

Outputs:
- JSON report with per-case details
- Markdown summary report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Iterable

import requests
from dotenv import load_dotenv

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from shared.py.clients.ollama_client import OllamaEmbeddings

_VN_STOPWORDS = {
    "và", "là", "của", "cho", "trong", "được", "không", "cần", "bạn", "mình", "nếu",
    "thì", "đã", "với", "các", "này", "kia", "ở", "để", "khi", "theo", "vui", "lòng",
    "nhé", "ạ", "hay", "lại", "nên", "rồi", "từ", "đến", "về", "có", "một", "những",
}

_LEAK_PATTERNS = [
    r"search_faq_tool",
    r"\{\s*\"name\"\s*:\s*\"search_faq_tool\"",
    r"```json",
    r"tool_calls",
    r"是否有调用",
]


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a)
    b = list(b)
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _tokenize(text: str) -> set[str]:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return {t for t in text.split() if len(t) >= 3 and t not in _VN_STOPWORDS}


def _keyword_coverage(reference: str, answer: str) -> float:
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 0.0
    # Limit to top frequent-like tokens by preserving first-seen order from reference
    seen = []
    for token in re.findall(r"\w+", reference.lower(), flags=re.UNICODE):
        if token in ref_tokens and token not in seen:
            seen.append(token)
        if len(seen) >= 24:
            break
    target = set(seen) if seen else ref_tokens
    ans_tokens = _tokenize(answer)
    return len(target.intersection(ans_tokens)) / max(1, len(target))


def _readability_ok(answer: str) -> bool:
    answer = (answer or "").strip()
    if len(answer) < 60:
        return False
    lowered = answer.lower()
    return not any(re.search(p, lowered) for p in _LEAK_PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate bot answer quality against ticket-derived references")
    parser.add_argument("--dataset", default="reports/eval/ticket_eval_cases.json", help="Input dataset JSON")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1/ask", help="Ask API endpoint")
    parser.add_argument("--ollama-url", default=None, help="Override OLLAMA_BASE_URL for embedding judge")
    parser.add_argument("--embedding-model", default="bge-m3", help="Embedding model for semantic similarity")
    parser.add_argument("--limit", type=int, default=40, help="Max cases to evaluate")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between requests to avoid overload")
    parser.add_argument("--out-json", default="reports/eval/bot_eval_report.json", help="Detailed JSON report")
    parser.add_argument("--out-md", default="reports/eval/bot_eval_report.md", help="Markdown summary report")
    args = parser.parse_args()

    load_dotenv("/home/phungkien/ehc_ai_mvp/.env")

    with open(args.dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)
    cases = list(ds.get("cases", []))[: args.limit]
    if not cases:
        raise RuntimeError("Dataset has no cases")

    ollama_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedder = OllamaEmbeddings(base_url=ollama_url, model=args.embedding_model, timeout=120)

    details = []
    overall_scores = []
    source_hits = 0
    readability_hits = 0

    for i, case in enumerate(cases, start=1):
        query = case["query"]
        expected_issue_id = str(case["expected_issue_id"])
        reference = case["reference_answer"]

        started = time.time()
        response = requests.post(args.api_url, json={"query": query}, timeout=240)
        response.raise_for_status()
        payload = response.json()
        answer = (payload.get("answer") or "").strip()

        sources = payload.get("sources") or []
        top_issue_id = str(sources[0].get("issue_id", "")) if sources else ""
        source_hit = top_issue_id == expected_issue_id
        if source_hit:
            source_hits += 1

        ans_vec = embedder.embed_query(answer)
        ref_vec = embedder.embed_query(reference)
        semantic_similarity = _cosine(ans_vec, ref_vec)

        keyword_coverage = _keyword_coverage(reference, answer)
        readability_ok = _readability_ok(answer)
        if readability_ok:
            readability_hits += 1

        # Weighted score: grounded source + semantic + coverage + readability
        overall = (
            0.35 * (1.0 if source_hit else 0.0)
            + 0.35 * semantic_similarity
            + 0.2 * keyword_coverage
            + 0.1 * (1.0 if readability_ok else 0.0)
        )
        overall_scores.append(overall)

        details.append(
            {
                "case_id": case["case_id"],
                "query": query,
                "expected_issue_id": expected_issue_id,
                "predicted_issue_id": top_issue_id,
                "source_hit": source_hit,
                "semantic_similarity": round(float(semantic_similarity), 4),
                "keyword_coverage": round(float(keyword_coverage), 4),
                "readability_ok": readability_ok,
                "overall_score": round(float(overall), 4),
                "latency_ms": round((time.time() - started) * 1000, 1),
                "answer_preview": answer[:320],
                "reference_preview": reference[:320],
            }
        )

        if args.sleep_ms > 0 and i < len(cases):
            time.sleep(args.sleep_ms / 1000)

    mean_overall = statistics.mean(overall_scores) if overall_scores else 0.0
    p50_overall = statistics.median(overall_scores) if overall_scores else 0.0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "api_url": args.api_url,
        "cases_evaluated": len(details),
        "source_hit_rate": round(source_hits / max(1, len(details)), 4),
        "readability_rate": round(readability_hits / max(1, len(details)), 4),
        "overall_mean": round(float(mean_overall), 4),
        "overall_p50": round(float(p50_overall), 4),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Bot Evaluation Report\n\n")
        f.write(f"- Generated at: {summary['generated_at']}\n")
        f.write(f"- Cases evaluated: {summary['cases_evaluated']}\n")
        f.write(f"- Source hit rate: {summary['source_hit_rate']:.2%}\n")
        f.write(f"- Readability rate: {summary['readability_rate']:.2%}\n")
        f.write(f"- Overall mean score: {summary['overall_mean']:.4f}\n")
        f.write(f"- Overall P50 score: {summary['overall_p50']:.4f}\n\n")

        worst = sorted(details, key=lambda x: x["overall_score"])[:10]
        f.write("## Lowest-Scoring Cases\n\n")
        for item in worst:
            f.write(f"### {item['case_id']}\n")
            f.write(f"- Query: {item['query']}\n")
            f.write(f"- Expected issue: {item['expected_issue_id']}\n")
            f.write(f"- Predicted issue: {item['predicted_issue_id']}\n")
            f.write(f"- Source hit: {item['source_hit']}\n")
            f.write(f"- Semantic similarity: {item['semantic_similarity']}\n")
            f.write(f"- Keyword coverage: {item['keyword_coverage']}\n")
            f.write(f"- Readability OK: {item['readability_ok']}\n")
            f.write(f"- Overall score: {item['overall_score']}\n")
            f.write(f"- Answer preview: {item['answer_preview']}\n\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed report: {args.out_json}")
    print(f"Saved markdown summary: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
