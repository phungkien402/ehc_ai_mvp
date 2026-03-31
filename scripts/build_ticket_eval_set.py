#!/usr/bin/env python3
"""Build evaluation cases from Redmine FAQ tickets.

Output JSON schema:
{
  "generated_at": "...",
  "project": "ehcfaq",
  "total_cases": N,
  "cases": [
    {
      "case_id": "ticket_12345",
      "query": "...",
      "expected_issue_id": "12345",
      "reference_answer": "...",
      "subject": "...",
      "source_url": "...",
      "attachment_urls": ["..."]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from pipelines.ingestion.app.sources.redmine_client import RedmineClient


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _subject_to_query(subject: str) -> str:
    s = _clean_text(subject).rstrip("?.! ")
    lowered = s.lower()
    if lowered.startswith(("tại sao", "tai sao", "vì sao", "vi sao", "sao ")):
        return s + "?"
    if lowered.startswith(("cách ", "lam sao", "làm sao", "hướng dẫn", "huong dan")):
        return s + "?"
    return f"Cho mình hỏi {s.lower()} được không?"


def _build_reference_answer(subject: str, description: str, max_chars: int) -> str:
    subject = _clean_text(subject)
    description = _clean_text(description)
    combined = f"{subject}\n\n{description}".strip()
    if len(combined) <= max_chars:
        return combined
    return combined[:max_chars].rstrip() + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bot evaluation dataset from Redmine tickets")
    parser.add_argument("--project", default="ehcfaq", help="Redmine project identifier")
    parser.add_argument("--limit", type=int, default=80, help="Max number of tickets to include")
    parser.add_argument("--max-reference-chars", type=int, default=1200, help="Max chars per reference answer")
    parser.add_argument(
        "--output",
        default="reports/eval/ticket_eval_cases.json",
        help="Output path for generated dataset JSON",
    )
    args = parser.parse_args()

    load_dotenv("/home/phungkien/ehc_ai_mvp/.env")

    redmine_url = os.getenv("REDMINE_URL", "").strip()
    redmine_api_key = os.getenv("REDMINE_API_KEY", "").strip()
    if not redmine_url or not redmine_api_key:
        raise RuntimeError("Missing REDMINE_URL or REDMINE_API_KEY in .env")

    client = RedmineClient(base_url=redmine_url, api_key=redmine_api_key)
    faqs = client.fetch_faq_issues(project_key=args.project, status_ids=["*"])
    if not faqs:
        raise RuntimeError("No FAQ tickets fetched from Redmine")

    cases = []
    for faq in faqs[: args.limit]:
        ref = _build_reference_answer(faq.subject, faq.description, args.max_reference_chars)
        if len(ref) < 60:
            continue
        cases.append(
            {
                "case_id": f"ticket_{faq.issue_id}",
                "query": _subject_to_query(faq.subject),
                "expected_issue_id": str(faq.issue_id),
                "reference_answer": ref,
                "subject": faq.subject,
                "source_url": faq.url,
                "attachment_urls": faq.attachment_urls,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": args.project,
        "total_cases": len(cases),
        "cases": cases,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(cases)} eval cases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
