#!/usr/bin/env python3
"""Stream raw vLLM output — shows thinking in dim and extracted answer in bold."""

import json
import os
import re
import sys
import urllib.request
import urllib.error

GREY  = "\033[2;90m"
BOLD  = "\033[1m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith(name + "="):
                    val = line[len(name) + 1:].strip().strip('"').strip("'")
                    os.environ[name] = val
                    return val
    except FileNotFoundError:
        pass
    return default


VLLM_URL   = _env("VLLM_LLM_URL") or _env("VLLM_BASE_URL", "http://localhost:8080")
VLLM_MODEL = _env("VLLM_LLM_MODEL", "Qwen/Qwen3.5-4B")
VLLM_KEY   = _env("VLLM_API_KEY", "EMPTY")

SYSTEM = "Bạn là trợ lý thân thiện. Trả lời tự nhiên, ngắn gọn bằng tiếng Việt."


def _extract_answer(raw: str, reasoning: str) -> tuple[str, str]:
    """Return (thinking_text, final_answer)."""
    if reasoning:
        return reasoning, re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("Thinking Process:") or "\nThinking Process:" in raw:
        quotes = re.findall(r'"([^"]{5,})"', raw)
        answer = quotes[-1].strip() if quotes else ""
        if not answer:
            for para in reversed(raw.split("\n\n")):
                para = para.strip()
                if para and not re.match(r"^[\*\-#\d]", para) and len(para) > 15:
                    answer = para.splitlines()[-1].strip()
                    break
        return raw, answer
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL).strip()
    return "", stripped


def stream_chat(user_msg: str) -> None:
    url = f"{VLLM_URL.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "stream": True,
        "temperature": 0.6,
        "max_tokens": 1024,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_KEY}",
        },
        method="POST",
    )

    print(f"{GREY}model: {VLLM_MODEL}  →  {url}{RESET}")
    print(f"{GREY}user : {user_msg!r}{RESET}\n")

    chunks_content: list[str] = []
    chunks_reasoning: list[str] = []

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                r = delta.get("reasoning_content") or ""
                c = delta.get("content") or ""
                if r:
                    chunks_reasoning.append(r)
                    # stream reasoning in grey immediately
                    print(f"{GREY}{r}{RESET}", end="", flush=True)
                if c:
                    chunks_content.append(c)
                    # stream content as it arrives — we'll re-print after
                    if not chunks_reasoning and not (
                        "".join(chunks_content).startswith("Thinking Process:")
                        or "\nThinking Process:" in "".join(chunks_content)
                    ):
                        # no thinking detected yet — print live
                        print(f"{c}", end="", flush=True)

    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"\n{GREY}HTTP {e.code}: {body}{RESET}")
        return
    except Exception as exc:
        print(f"\nError: {exc}")
        return

    raw     = "".join(chunks_content)
    reasoning = "".join(chunks_reasoning)

    thinking, answer = _extract_answer(raw, reasoning)

    if thinking:
        # re-display thinking in grey (clear and reprint)
        if not chunks_reasoning:  # inline thinking case — wasn't printed live
            print(f"\r{GREY}{thinking}{RESET}", flush=True)
        else:
            print()  # newline after streamed reasoning

    print(f"\n{CYAN}{'─'*50}{RESET}")
    print(f"{GREEN}[ANSWER]{RESET} {BOLD}{answer}{RESET}")
    print(f"{CYAN}{'─'*50}{RESET}")
    print(f"{GREY}think: {len(thinking)} chars  answer: {len(answer)} chars{RESET}\n")


def main() -> None:
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
    else:
        print("EHC test_llm — direct vLLM stream")
        print("Ctrl+C to quit\n")
        while True:
            try:
                msg = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                return
            if msg:
                stream_chat(msg)

    stream_chat(msg)


if __name__ == "__main__":
    main()
