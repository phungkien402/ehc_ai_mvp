"""Quick token throughput test using vLLM endpoint — output saved to /tmp/token_speed_result.txt"""

import io
import json
import sys
import time

import requests

OUTPUT_FILE = "/tmp/token_speed_result.txt"
_buf = io.StringIO()

def p(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, file=_buf, **kwargs)

VLLM_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-4B"

SYSTEM = (
    "/no_think\n"
    "Bạn là trợ lý helpdesk EHC, hỗ trợ nhân viên y tế dùng phần mềm HIS.\n"
    "Trả lời tự nhiên, thân thiện, ngắn gọn bằng tiếng Việt. Không dùng Markdown."
)

USER = (
    "Câu hỏi: Hướng dẫn mình gộp mã bệnh nhân trong hệ thống HIS\n\n"
    "Thông tin tra cứu:\n"
    "Chunk 1:\nsource_type: faq\nsource_id: 34001\ntitle: Gộp mã bệnh nhân\n"
    "content:\nVào Hành chính -> Quản lý bệnh nhân -> Tìm bệnh nhân trùng -> "
    "Chọn bệnh nhân cần gộp -> Nhấn Gộp -> Xác nhận.\n"
    "Lưu ý: chỉ gộp được khi cả 2 mã chưa có hóa đơn thanh toán.\n\n"
    "Dựa vào thông tin trên, trả lời câu hỏi của người dùng."
)

MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
]

def test_non_stream():
    p("=== Non-streaming test ===")
    t0 = time.time()
    try:
        resp = requests.post(VLLM_URL, json={
            "model": MODEL,
            "messages": MESSAGES,
            "max_tokens": 300,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=60)
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0

        p(f"Prompt tokens    : {prompt_tokens}")
        p(f"Completion tokens: {completion_tokens}")
        p(f"Total time       : {elapsed:.2f}s")
        p(f"Tokens/second    : {tps:.1f} tok/s")
        p(f"\nAnswer:\n{content}")
    except Exception as e:
        p(f"ERROR: {e}")


def test_stream():
    p("\n=== Streaming test ===")
    t0 = time.time()
    first_token_time = None
    chunks = []

    try:
        with requests.post(VLLM_URL, json={
            "model": MODEL,
            "messages": MESSAGES,
            "max_tokens": 300,
            "temperature": 0.1,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                chunk_data = json.loads(payload)
                delta = chunk_data["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    if first_token_time is None:
                        first_token_time = time.time()
                    chunks.append(delta)

        elapsed = time.time() - t0
        ttft = (first_token_time - t0) if first_token_time else elapsed
        gen_time = elapsed - ttft
        tps = len(chunks) / gen_time if gen_time > 0 else 0

        p(f"Time to first token: {ttft:.2f}s")
        p(f"Total time         : {elapsed:.2f}s")
        p(f"Chunks received    : {len(chunks)}")
        p(f"Generation speed   : {tps:.1f} chunks/s (≈ tok/s)")
        p(f"\nAnswer:\n{''.join(chunks)}")
    except Exception as e:
        p(f"ERROR: {e}")


if __name__ == "__main__":
    test_non_stream()
    test_stream()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(_buf.getvalue())

    print(f"\nResults saved to {OUTPUT_FILE}")
