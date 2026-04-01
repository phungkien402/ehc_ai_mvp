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
import time
from logging.handlers import RotatingFileHandler
from urllib.parse import unquote, urlparse

import httpx
from dotenv import load_dotenv
from telegram import InputFile, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import NetworkError as TelegramNetworkError
from telegram.error import RetryAfter, TimedOut as TelegramTimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("HELPDESK_API_URL", "http://localhost:8000/api/v1/ask")
API_TIMEOUT_SECONDS = int(os.getenv("TELEGRAM_API_TIMEOUT_SECONDS", "240"))
API_RETRIES = int(os.getenv("TELEGRAM_API_RETRIES", "1"))
RETRY_BACKOFF_SECONDS = float(os.getenv("TELEGRAM_API_RETRY_BACKOFF_SECONDS", "1.5"))
REDMINE_API_KEY = os.getenv("REDMINE_API_KEY", "")
REDMINE_URL = os.getenv("REDMINE_URL", "")
MAX_IMAGE_SEND = int(os.getenv("TELEGRAM_MAX_IMAGE_SEND", "3"))
TELEGRAM_STREAM_ENABLED = os.getenv("TELEGRAM_STREAM_ENABLED", "true").lower() == "true"
TELEGRAM_STREAM_CHUNK_WORDS = int(os.getenv("TELEGRAM_STREAM_CHUNK_WORDS", "8"))
TELEGRAM_STREAM_EDIT_INTERVAL_SECONDS = float(os.getenv("TELEGRAM_STREAM_EDIT_INTERVAL_SECONDS", "0.35"))
TELEGRAM_SEND_RETRIES = int(os.getenv("TELEGRAM_SEND_RETRIES", "2"))
TELEGRAM_SEND_RETRY_BACKOFF_SECONDS = float(
    os.getenv("TELEGRAM_SEND_RETRY_BACKOFF_SECONDS", "1.0")
)
TELEGRAM_CONNECT_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_CONNECT_TIMEOUT_SECONDS", "10"))
TELEGRAM_READ_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_READ_TIMEOUT_SECONDS", "30"))
TELEGRAM_WRITE_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_WRITE_TIMEOUT_SECONDS", "30"))
TELEGRAM_POOL_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_POOL_TIMEOUT_SECONDS", "10"))

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


# Keep last uploaded image per chat so users can send photo first, then ask text follow-up.
RECENT_IMAGE_TTL_SECONDS = int(os.getenv("TELEGRAM_RECENT_IMAGE_TTL_SECONDS", "300"))
_RECENT_IMAGE_BY_CHAT: dict[str, tuple[float, bytes]] = {}


async def _telegram_with_retry(op_name: str, op_factory):
    """Run Telegram API call with retry for transient timeout/network errors."""
    for attempt in range(TELEGRAM_SEND_RETRIES + 1):
        try:
            return await op_factory()
        except RetryAfter as e:
            if attempt >= TELEGRAM_SEND_RETRIES:
                raise
            wait_s = float(getattr(e, "retry_after", 1) or 1)
            logger.warning("%s rate-limited; retry in %.1fs", op_name, wait_s)
            await asyncio.sleep(wait_s)
        except (TelegramTimedOut, TelegramNetworkError) as e:
            if attempt >= TELEGRAM_SEND_RETRIES:
                raise
            wait_s = TELEGRAM_SEND_RETRY_BACKOFF_SECONDS * (attempt + 1)
            logger.warning(
                "%s transient Telegram error (%s), retry %d/%d in %.1fs",
                op_name,
                e,
                attempt + 1,
                TELEGRAM_SEND_RETRIES,
                wait_s,
            )
            await asyncio.sleep(wait_s)


async def _safe_reply_text(
    update: Update,
    text: str,
    parse_mode: str | None = None,
    disable_web_page_preview: bool | None = None,
) -> bool:
    try:
        await _telegram_with_retry(
            "reply_text",
            lambda: update.message.reply_text(
                text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            ),
        )
        return True
    except Exception as e:
        logger.error("reply_text failed after retries: %s", e)
        return False


async def _safe_reply_photo(update: Update, photo, **kwargs) -> bool:
    try:
        await _telegram_with_retry(
            "reply_photo",
            lambda: update.message.reply_photo(photo=photo, **kwargs),
        )
        return True
    except Exception as e:
        logger.error("reply_photo failed after retries: %s", e)
        return False


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


def _remember_recent_image(chat_id: str, image_bytes: bytes) -> None:
    if not chat_id or not image_bytes:
        return
    _RECENT_IMAGE_BY_CHAT[chat_id] = (time.time(), image_bytes)


def _get_recent_image(chat_id: str) -> bytes | None:
    item = _RECENT_IMAGE_BY_CHAT.get(chat_id)
    if not item:
        return None
    ts, image_bytes = item
    if (time.time() - ts) > RECENT_IMAGE_TTL_SECONDS:
        _RECENT_IMAGE_BY_CHAT.pop(chat_id, None)
        return None
    return image_bytes


def _looks_like_image_followup(query: str) -> bool:
    lowered = (query or "").lower().strip()
    if not lowered:
        return False
    patterns = [
        r"\bl[oô]i\s+n[aà]y\b",
        r"\bc[aá]i\s+n[aà]y\b",
        r"\b[aả]nh\s+n[aà]y\b",
        r"\bh[iì]nh\s+n[aà]y\b",
        r"\bgi[uú]p\s+m[iì]nh\b",
    ]
    return any(re.search(p, lowered) for p in patterns)


async def _ask_api(query: str, image_bytes: bytes | None = None, user_id: str | None = None,
                   channel: str | None = None) -> dict:
    """Gọi FastAPI /ask và trả về response dict."""
    payload: dict = {"query": query}
    if image_bytes:
        payload["image_base64"] = base64.b64encode(image_bytes).decode()
    if user_id:
        payload["user_id"] = user_id
    if channel:
        payload["channel"] = channel

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
    # Ticket citation disabled
    # if sources:
    #     src = sources[0]
    #     issue_id = src.get("issue_id", "")
    #     url = src.get("url", "")
    #     if url:
    #         safe_url = html.escape(str(url), quote=True)
    #         lines.append(f"\nNguồn: <a href=\"{safe_url}\">Ticket #{html.escape(str(issue_id))}</a>")
    #     elif issue_id:
    #         lines.append(f"\nNguồn: Ticket #{html.escape(str(issue_id))}")
    #
    #     related = []
    #     for s in sources[1:4]:
    #         rel_issue_id = s.get("issue_id", "")
    #         rel_url = s.get("url", "")
    #         if not rel_issue_id:
    #             continue
    #         if rel_url:
    #             safe_rel_url = html.escape(str(rel_url), quote=True)
    #             related.append(
    #                 f"<a href=\"{safe_rel_url}\">#{html.escape(str(rel_issue_id))}</a>"
    #             )
    #         else:
    #             related.append(f"#{html.escape(str(rel_issue_id))}")
    #     if related:
    #         lines.append(f"Ticket gần nghĩa: {', '.join(related)}")

    return "\n".join(lines)


def _format_reply_plain(data: dict) -> str:
    """Render plain-text reply for progressive Telegram message edits."""
    answer = _clean_answer(data.get("answer", "").strip())
    error = data.get("error")
    if error and not answer:
        return f"Lỗi: {error}"
    return answer or "(Không có nội dung trả lời)"


def _progressive_chunks(text: str, words_per_chunk: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return [text or ""]
    step = max(1, words_per_chunk)
    chunks: list[str] = []
    for i in range(step, len(words) + step, step):
        chunks.append(" ".join(words[:i]))
    if chunks[-1] != " ".join(words):
        chunks.append(" ".join(words))
    return chunks


async def _stream_reply_text(update: Update, full_text: str) -> bool:
    """Simulate streaming UX by editing one Telegram message progressively."""
    if not TELEGRAM_STREAM_ENABLED:
        return await _safe_reply_text(update, full_text)

    # Keep payload size within Telegram limits for text messages.
    final_text = (full_text or "").strip()[:3900]
    if not final_text:
        final_text = "(Không có nội dung trả lời)"

    placeholder = "..."
    try:
        msg = await _telegram_with_retry(
            "reply_text_stream_placeholder",
            lambda: update.message.reply_text(placeholder),
        )
    except Exception as e:
        logger.error("stream placeholder send failed: %s", e)
        return False

    try:
        chunks = _progressive_chunks(final_text, TELEGRAM_STREAM_CHUNK_WORDS)
        for idx, chunk in enumerate(chunks):
            await _telegram_with_retry(
                "edit_message_stream",
                lambda c=chunk: msg.edit_text(c, disable_web_page_preview=True),
            )
            if idx < len(chunks) - 1:
                await asyncio.sleep(TELEGRAM_STREAM_EDIT_INTERVAL_SECONDS)
        return True
    except Exception as e:
        logger.error("stream edit failed: %s", e)
        return False


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or "attachment.jpg"
    return unquote(name)


def _is_redmine_attachment(url: str) -> bool:
    """Return True if URL is a Redmine attachment that Telegram cannot fetch directly."""
    if REDMINE_URL and url.startswith(REDMINE_URL):
        return True
    return "/redmine/attachments/download/" in url


async def _send_image_attachments(update: Update, image_urls: list[str]) -> None:
    """Send image attachments directly to Telegram chat.

    Strategy:
    1) Try sending URL directly, unless URL is a Redmine attachment (Telegram can't
       reach internal Redmine — go straight to bytes download in that case).
    2) Download image with optional Redmine API key and upload bytes.
    3) If all fail, send text fallback with links.
    """

    if not image_urls:
        return

    sent_count = 0
    failed_urls: list[str] = []

    for url in image_urls[:MAX_IMAGE_SEND]:
        # First attempt: direct URL send — skip for Redmine internal attachments
        # because Telegram's servers cannot reach co.ehc.vn and would waste ~6s on retries.
        if _is_redmine_attachment(url):
            logger.debug("Skipping direct URL send for Redmine attachment: %s", url)
        else:
            ok = await _safe_reply_photo(update, url)
            if ok:
                sent_count += 1
                continue
            logger.warning("Direct URL send failed for %s", url)

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
            ok = await _safe_reply_photo(
                update,
                InputFile(io.BytesIO(resp.content), filename=filename),
            )
            if not ok:
                raise RuntimeError("Failed to send image attachment to Telegram")
            sent_count += 1
        except Exception as fetch_err:
            logger.error("Attachment send failed for %s: %s", url, fetch_err)
            failed_urls.append(url)

    if sent_count == 0 and failed_urls:
        fallback_lines = ["⚠️ Không gửi trực tiếp được ảnh đính kèm. Bạn mở tạm các link sau:"]
        for url in failed_urls:
            fallback_lines.append(f"- {url}")
        await _safe_reply_text(update, "\n".join(fallback_lines))


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _safe_reply_text(
        update,
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

    try:
        await _telegram_with_retry(
            "send_action_text",
            lambda: update.message.chat.send_action(ChatAction.TYPING),
        )
    except Exception as e:
        logger.warning("send_action failed (text): %s", e)

    try:
        chat_id = str(getattr(update.effective_chat, "id", ""))
        followup_image = _get_recent_image(chat_id) if _looks_like_image_followup(query) else None
        data = await _ask_api(query, followup_image, user_id=chat_id, channel="telegram")
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
        reply_plain = _format_reply_plain(data)
        if TELEGRAM_STREAM_ENABLED and not data.get("error") and reply_plain.strip():
            ok = await _stream_reply_text(update, reply_plain)
            if not ok:
                ok = await _safe_reply_text(
                    update,
                    reply,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        else:
            ok = await _safe_reply_text(
                update,
                reply,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        if not ok:
            return
        await _send_image_attachments(update, data.get("image_urls", []))
    except httpx.ReadTimeout as e:
        logger.error("API timeout: %s", e)
        await _safe_reply_text(
            update,
            "⚠️ Hệ thống đang xử lý chậm. Vui lòng thử lại sau ít phút."
        )
    except httpx.ConnectError as e:
        logger.error("API connect error: %s", e)
        await _safe_reply_text(
            update,
            "⚠️ Không kết nối được tới máy chủ trợ lý. Vui lòng thử lại sau."
        )
    except httpx.HTTPStatusError as e:
        logger.error("API status error: %s", e)
        await _safe_reply_text(
            update,
            "⚠️ Máy chủ đang bận hoặc gặp lỗi phản hồi. Vui lòng thử lại sau."
        )
    except httpx.HTTPError as e:
        logger.error("API call failed: %s", e)
        await _safe_reply_text(
            update,
            "⚠️ Không kết nối được tới server. Vui lòng thử lại sau."
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý ảnh + caption (dùng caption làm query, ảnh để OCR)."""
    caption = (update.message.caption or "").strip()
    query = caption if caption else ""

    try:
        await _telegram_with_retry(
            "send_action_photo",
            lambda: update.message.chat.send_action(ChatAction.TYPING),
        )
    except Exception as e:
        logger.warning("send_action failed (photo): %s", e)

    try:
        # Tải ảnh kích thước cao nhất
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        chat_id = str(getattr(update.effective_chat, "id", ""))
        _remember_recent_image(chat_id, bytes(image_bytes))

        # If user only sends photo without caption, ask a clarifying text while
        # keeping image in short-term memory for the next text message.
        if not query:
            await _safe_reply_text(
                update,
                "Mình đã nhận ảnh rồi. Bạn nhắn thêm mô tả lỗi hoặc câu hỏi (ví dụ: 'giúp mình lỗi này với'), mình sẽ phân tích theo ảnh vừa gửi.",
            )
            return

        data = await _ask_api(query, bytes(image_bytes), user_id=chat_id, channel="telegram")
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
        reply_plain = _format_reply_plain(data)
        if TELEGRAM_STREAM_ENABLED and not data.get("error") and reply_plain.strip():
            ok = await _stream_reply_text(update, reply_plain)
            if not ok:
                ok = await _safe_reply_text(
                    update,
                    reply,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        else:
            ok = await _safe_reply_text(
                update,
                reply,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        if not ok:
            return
        await _send_image_attachments(update, data.get("image_urls", []))
    except httpx.ReadTimeout as e:
        logger.error("API timeout: %s", e)
        await _safe_reply_text(
            update,
            "⚠️ Hệ thống đang xử lý chậm. Vui lòng thử lại sau ít phút."
        )
    except httpx.ConnectError as e:
        logger.error("API connect error: %s", e)
        await _safe_reply_text(
            update,
            "⚠️ Không kết nối được tới máy chủ trợ lý. Vui lòng thử lại sau."
        )
    except httpx.HTTPStatusError as e:
        logger.error("API status error: %s", e)
        await _safe_reply_text(
            update,
            "⚠️ Máy chủ đang bận hoặc gặp lỗi phản hồi. Vui lòng thử lại sau."
        )
    except httpx.HTTPError as e:
        logger.error("API call failed: %s", e)
        await _safe_reply_text(
            update,
            "⚠️ Không kết nối được tới server. Vui lòng thử lại sau."
        )


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled Telegram update error: %s", context.error)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN chưa được đặt trong .env")

    request = HTTPXRequest(
        connect_timeout=TELEGRAM_CONNECT_TIMEOUT_SECONDS,
        read_timeout=TELEGRAM_READ_TIMEOUT_SECONDS,
        write_timeout=TELEGRAM_WRITE_TIMEOUT_SECONDS,
        pool_timeout=TELEGRAM_POOL_TIMEOUT_SECONDS,
    )
    get_updates_request = HTTPXRequest(
        connect_timeout=TELEGRAM_CONNECT_TIMEOUT_SECONDS,
        read_timeout=TELEGRAM_READ_TIMEOUT_SECONDS,
        write_timeout=TELEGRAM_WRITE_TIMEOUT_SECONDS,
        pool_timeout=TELEGRAM_POOL_TIMEOUT_SECONDS,
    )

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .get_updates_request(get_updates_request)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(_on_error)

    logger.info("Bot đang chạy (polling)... Ctrl+C để dừng.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
