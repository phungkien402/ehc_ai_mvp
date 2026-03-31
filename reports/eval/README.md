Evaluation Workflow

1) Build test cases from real Redmine tickets:
python3 scripts/build_ticket_eval_set.py --project ehcfaq --limit 80

2) Run bot evaluation:
python3 scripts/run_bot_eval.py --dataset reports/eval/ticket_eval_cases.json --limit 40

Outputs:
- reports/eval/ticket_eval_cases.json
- reports/eval/bot_eval_report.json
- reports/eval/bot_eval_report.md

Score components:
- source_hit (expected issue_id match)
- semantic_similarity (embedding cosine: answer vs ticket reference)
- keyword_coverage (important ticket terms appearing in answer)
- readability_ok (no raw tool leakage, enough answer length)
