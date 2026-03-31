"""Telegram bot frontend cho EHC AI Helpdesk.

Dùng polling (không cần webhook/domain/SSL).
Kết nối tới FastAPI server tại http://localhost:8000.

Chạy: venv/bin/python3 apps/telegram_bot/bot.py
"""

import asyncio
import base64
import html
import io
import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from urllib.parse import unquote, urlparse

import httpx
from dotenv import load_dotenv
from telegram import InputFile, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("HELPDESK_API_URL", "http://localhost:8000/api/v1/ask")
API_TIMEOUT_SECONDS = int(os.getenv("TELEGRAM_API_TIMEOUT_SECONDS", "240"))
API_RETRIES = int(os.getenv("TELEGRAM_API_RETRIES", "1"))
RETRY_BACKOFF_SECONDS = float(os.getenv("TELEGRAM_API_RETRY_BACKOFF_SECONDS", "1.5"))
REDMINE_API_KEY = os.getenv("REDMINE_API_KEY", "")
MAX_IMAGE_SEND = int(os.getenv("TELEGRAM_MAX_IMAGE_SEND", "3"))

# Dedicated bot log file for easier debugging in dashboard/log tail.
log_level_name = os.getenv("BOT_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

os.makedirs("logs", exist_ok=True)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

file_handler = RotatingFileHandler("logs/bot.log", maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "Cao"
    if score >= 0.55:
        return "Trung bình"
    return "Thấp"


def _clean_answer(answer: str) -> str:
    """Remove trailing metadata footer inserted by backend, keep answer body only."""
    lines = [line.rstrip() for line in answer.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith("_(tham khảo ticket") or line_lower.startswith("(tham khao ticket"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _to_html_multiline(text: str) -> str:
    # Telegram HTML parse mode does not support <br>; keep real newlines.
    text = html.escape(text.strip())
    return re.sub(r"\n{3,}", "\n\n", text)


async def _ask_api(query: str, image_bytes: bytes | None = None) -> dict:
    """Gọi FastAPI /ask và trả về response dict."""
    payload: dict = {"query": query}
    if image_bytes:
        payload["image_base64"] = base64.b64encode(image_bytes).decode()

    last_error: Exception | None = None
    for attempt in range(1, API_RETRIES + 2):
        try:
            timeout = httpx.Timeout(API_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(API_URL, json=payload)
                resp.raise_for_status()
                return resp.json()
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            last_error = e
            if attempt > API_RETRIES:
                break
            delay = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "API transient error on attempt %s/%s: %s. Retrying in %.1fs",
                attempt,
                API_RETRIES + 1,
                e,
                delay,
            )
            await asyncio.sleep(delay)
    if last_error:
        raise last_error
    raise RuntimeError("Unknown API call failure")


def _format_reply(data: dict) -> str:
    """Render user-facing reply in natural chat style (answer only)."""
    answer = _clean_answer(data.get("answer", "").strip())
    sources = data.get("sources", [])
    error = data.get("error")

    if error and not answer:
        return f"⚠️ <b>Lỗi:</b> {html.escape(str(error))}"

    lines = [_to_html_multiline(answer)]
    if sources:
        src = sources[0]
        issue_id = src.get("issue_id", "")
        url = src.get("url", "")
        if url:
            safe_url = html.escape(str(url), quote=True)
            lines.append(f"\nNguồn: <a href=\"{safe_url}\">Ticket #{html.escape(str(issue_id))}</a>")
        elif issue_id:
            lines.append(f"\nNguồn: Ticket #{html.escape(str(issue_id))}")

    return "\n".join(lines)


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or "attachment.jpg"
    return unquote(name)


async def _send_image_attachments(update: Update, image_urls: list[str]) -> None:
    """Send image attachments directly to Telegram chat.

    Strategy:
    1) Try sending URL directly (Telegram fetches remote image).
    2) If failed, download image with optional Redmine API key and upload bytes.
    3) If all fail, send text fallback with links.
    """

    if not image_urls:
        return

    sent_count = 0
    failed_urls: list[str] = []

    for url in image_urls[:MAX_IMAGE_SEND]:
        # First attempt: direct URL send
        try:
            await update.message.reply_photo(photo=url)
            sent_count += 1
            continue
        except Exception as direct_err:
            logger.warning("Direct URL send failed for %s: %s", url, direct_err)

        # Second attempt: authenticated fetch then upload bytes
        headers = {"X-Redmine-API-Key": REDMINE_API_KEY} if REDMINE_API_KEY else {}
        try:
            timeout = httpx.Timeout(API_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                raise ValueError(f"Attachment is not an image content-type: {content_type}")

            filename = _filename_from_url(url)
            await update.message.reply_photo(
                photo=InputFile(io.BytesIO(resp.content), filename=filename)
            )
            sent_count += 1
        except Exception as fetch_err:
            logger.error("Attachment send failed for %s: %s", url, fetch_err)
            failed_urls.append(url)

    if sent_count == 0 and failed_urls:
        fallback_lines = ["⚠️ Không gửi trực tiếp được ảnh đính kèm. Bạn mở tạm các link sau:"]
        for url in failed_urls:
            fallback_lines.append(f"- {url}")
        await update.message.reply_text("\n".join(fallback_lines))


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Xin chào! Tôi là <b>EHC AI Helpdesk</b>.\n\n"
        "Hãy gõ câu hỏi về hệ thống EHC (hoặc gửi ảnh screenshot kèm câu hỏi), "
        "tôi sẽ tìm câu trả lời từ kho FAQ.\n\n"
        "Ví dụ: <i>kiểm tra tồn kho thuốc ở đâu?</i>",
        parse_mode=ParseMode.HTML,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text.strip()
    if not query:
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        data = await _ask_api(query)
        src = (data.get("sources") or [{}])[0]
        score = float(src.get("score", 0) or 0)
        logger.info(
            "BOT_REPLY_META|mode=text|query=%s|issue_id=%s|score=%.4f|confidence=%s|elapsed_ms=%.0f|images=%d|error=%s",
            query[:120],
            src.get("issue_id", ""),
            score,
            _confidence_label(score),
            float(data.get("execution_time_ms", 0) or 0),
            len(data.get("image_urls", []) or []),
            str(data.get("error") or ""),
        )
        reply = _format_reply(data)
        await update.message.reply_text(reply, parse_mode=ParseMode.HTML,
                                        disable_web_page_preview=True)
        await _send_image_attachments(update, data.get("image_urls", []))
    except httpx.ReadTimeout as e:
        logger.error("API timeout: %s", e)
        await update.message.reply_text(
            "⚠️ Hệ thống đang xử lý chậm. Vui lòng thử lại sau ít phút."
        )
    except httpx.ConnectError as e:
        logger.error("API connect error: %s", e)
        await update.message.reply_text(
            "⚠️ Không kết nối được tới máy chủ trợ lý. Vui lòng thử lại sau."
        )
    except httpx.HTTPStatusError as e:
        logger.error("API status error: %s", e)
        await update.message.reply_text(
            "⚠️ Máy chủ đang bận hoặc gặp lỗi phản hồi. Vui lòng thử lại sau."
        )
    except httpx.HTTPError as e:
        logger.error("API call failed: %s", e)
        await update.message.reply_text(
            "⚠️ Không kết nối được tới server. Vui lòng thử lại sau."
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý ảnh + caption (dùng caption làm query, ảnh để OCR)."""
    caption = (update.message.caption or "").strip()
    query = caption if caption else "Mô tả nội dung trong ảnh này"

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        # Tải ảnh kích thước cao nhất
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        data = await _ask_api(query, bytes(image_bytes))
        src = (data.get("sources") or [{}])[0]
        score = float(src.get("score", 0) or 0)
        logger.info(
            "BOT_REPLY_META|mode=photo|query=%s|issue_id=%s|score=%.4f|confidence=%s|elapsed_ms=%.0f|images=%d|error=%s",
            query[:120],
            src.get("issue_id", ""),
            score,
            _confidence_label(score),
            float(data.get("execution_time_ms", 0) or 0),
            len(data.get("image_urls", []) or []),
            str(data.get("error") or ""),
        )
        reply = _format_reply(data)
        await update.message.reply_text(reply, parse_mode=ParseMode.HTML,
                                        disable_web_page_preview=True)
        await _send_image_attachments(update, data.get("image_urls", []))
    except httpx.ReadTimeout as e:
        logger.error("API timeout: %s", e)
        await update.message.reply_text(
            "⚠️ Hệ thống đang xử lý chậm. Vui lòng thử lại sau ít phút."
        )
    except httpx.ConnectError as e:
        logger.error("API connect error: %s", e)
        await update.message.reply_text(
            "⚠️ Không kết nối được tới máy chủ trợ lý. Vui lòng thử lại sau."
        )
    except httpx.HTTPStatusError as e:
        logger.error("API status error: %s", e)
        await update.message.reply_text(
            "⚠️ Máy chủ đang bận hoặc gặp lỗi phản hồi. Vui lòng thử lại sau."
        )
    except httpx.HTTPError as e:
        logger.error("API call failed: %s", e)
        await update.message.reply_text(
            "⚠️ Không kết nối được tới server. Vui lòng thử lại sau."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN chưa được đặt trong .env")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot đang chạy (polling)... Ctrl+C để dừng.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
