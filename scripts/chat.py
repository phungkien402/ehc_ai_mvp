#!/usr/bin/env python3
"""Interactive terminal chat for testing the EHC AI helpdesk API."""

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/ask")
THINKING_LOG = "/tmp/ehc_thinking.log"
SESSION_ID = str(uuid.uuid4())
USER_ID = "terminal_test"


def _spinner(stop_event):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r  {frames[i % len(frames)]} thinking...")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * 20 + "\r")
    sys.stdout.flush()


def ask(query: str) -> dict:
    payload = json.dumps({
        "query": query,
        "channel": "terminal",
        "user_id": USER_ID,
        "session_id": SESSION_ID,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"error": f"HTTP {e.code}: {body}", "answer": ""}
    except Exception as e:
        return {"error": str(e), "answer": ""}


def main():
    print("EHC AI Helpdesk — terminal chat")
    print(f"API     : {API_URL}")
    print(f"Session : {SESSION_ID}")
    print(f"Thinking: tail -f {THINKING_LOG}")
    print("Type your message. Ctrl+C or 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "bye"):
            print("Bye!")
            break

        stop = threading.Event()
        spinner = threading.Thread(target=_spinner, args=(stop,), daemon=True)
        spinner.start()

        result = ask(query)

        stop.set()
        spinner.join()

        answer = result.get("answer", "").strip()
        error = result.get("error", "")
        sources = result.get("sources", [])
        ms = result.get("execution_time_ms", 0)

        if error and not answer:
            print(f"Bot [ERROR]: {error}\n")
            continue

        print(f"Bot: {answer}")

        if sources:
            src = sources[0]
            label = src.get("source_title") or (f"Ticket #{src.get('issue_id')}" if src.get("issue_id") else "")
            stype = src.get("source_type", "")
            if label:
                tag = f"[{stype}] " if stype else ""
                print(f"     Nguồn: {tag}{label}")

        print(f"     ({ms:.0f}ms)\n")


if __name__ == "__main__":
    main()
