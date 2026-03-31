#!/usr/bin/env python3
"""Call the standalone OCR endpoint with a local image file."""

import argparse
import base64
import json
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Test OCR against the API gateway.")
    parser.add_argument("image_path", help="Path to the local image file")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/ocr", help="OCR endpoint URL")
    parser.add_argument("--model", default=None, help="Optional OCR model override, e.g. qwen3.5:9b")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists() or not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    payload = {
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
    }
    if args.model:
        payload["model"] = args.model

    response = requests.post(args.url, json=payload, timeout=180)
    print(f"HTTP {response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)
    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
