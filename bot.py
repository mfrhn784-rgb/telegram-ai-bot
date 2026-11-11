import os
import io
import time
import tempfile
import logging
import threading
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)
from google import genai
from google.genai import types

# 1. Load Environment Variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")  # Optional: Set your Telegram ID to lock /stats

if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in .env file or environment variables!")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    raise ValueError("Missing GEMINI_API_KEY in .env file or environment variables!")

# 2. Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 3. Cloud Health Server (Keeps Render / Koyeb alive without OOM)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"AI Telegram Bot is active!")

    def log_message(self, format, *args):
        pass

def start_cloud_health_server():
    port = int(os.getenv("PORT", 0))
    if port:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"🌐 Cloud health check listening on port {port}")
        return True
    return False

# 4. Initialize Gemini AI Client with zero retry lag (attempts=1 for instant fallback)
ai_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1),
        timeout=25000,  # 25 second timeout
    ),
)

# High-IQ instruction with Adaptive Length, Follow-Up Memory, and Clean Plain Text
HIGH_IQ_SYSTEM_INSTRUCTION = (
    "You are an exceptionally intelligent, articulate, and helpful AI assistant.\n\n"
    "CORE PRINCIPLES:\n"
    "1. ADAPTIVE RESPONSE LENGTH (CRITICAL): Always match your answer length to the complexity of the question:\n"
    "   - Casual chats or simple factual questions (e.g., 'Hi', 'Who invented the lightbulb?', '5+7'): Reply naturally in 1 to 2 crisp, direct sentences.\n"
    "   - Standard questions or study doubts: Give a clear, medium-length explanation with key definitions and easy steps.\n"
    "   - Deep, complex, or 'explain in detail' prompts, or PDF/image reviews: Provide a thorough, well-structured, comprehensive breakdown.\n"
    "2. CONTEXT & FOLLOW-UPS: You remember this ongoing conversation. Naturally reference prior messages when the user asks follow-ups.\n"
    "3. CLEAN PLAIN TEXT (ZERO SYMBOLS): Telegram does not render markdown well. NEVER use symbols like #, ##, *, **, $, _, ~, `, or {}.\n"
    "   To organize longer answers, use neat line breaks, plain uppercase section labels (like OVERVIEW, DETAILS), or simple numbers (1, 2, 3).\n"
    "4. SCIENCE & REACTIONS: Explain reactions simply in plain words first, then clean keyboard formulas without confusing code symbols."
)

APP_CONFIG = types.GenerateContentConfig(
    system_instruction=HIGH_IQ_SYSTEM_INSTRUCTION,
    max_output_tokens=2048,
    temperature=0.6,
)

# Fastest, most reliable models first (avoids 503 lags on overloaded preview models)
MODELS_TO_TRY = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
]

# 5. Per-User Conversation Memory & Usage Analytics
chat_histories: dict[int, list[types.Content]] = {}
MAX_HISTORY_MESSAGES = 10

BOT_START_TIME = time.time()
usage_stats = {
    "unique_users": set(),
    "user_names": {},
    "text_count": 0,
    "photo_count": 0,
    "doc_count": 0,
    "total_tokens": 0,
    "recent_logs": deque(maxlen=10),
}


def record_activity(user_id: int, user_name: str, action_type: str, summary: str):
    """Record anonymized usage statistics and recent events."""
    usage_stats["unique_users"].add(user_id)
    usage_stats["user_names"][user_id] = user_name

    if action_type == "text":
        usage_stats["text_count"] += 1
    elif action_type == "photo":
        usage_stats["photo_count"] += 1
    elif action_type == "doc":
        usage_stats["doc_count"] += 1

    timestamp = datetime.now().strftime("%H:%M")
    short_summary = (summary[:30] + "...") if len(summary) > 30 else summary
    usage_stats["recent_logs"].append(f"[{timestamp}] {user_name} ({action_type}): {short_summary}")


def add_to_history(chat_id: int, role: str, parts: list):
    """Store message turn in memory while keeping RAM compact."""
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []

    content = types.Content(role=role, parts=parts)
    chat_histories[chat_id].append(content)

    if len(chat_histories[chat_id]) > MAX_HISTORY_MESSAGES:
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY_MESSAGES:]


def get_history(chat_id: int) -> list[types.Content]:
    """Retrieve existing conversation turns for this chat."""
    return chat_histories.get(chat_id, [])


async def generate_gemini_response(chat_id: int, current_parts: list) -> str:
    """Generate content using conversation history with instant model failover."""
    history = list(get_history(chat_id))
    current_content = types.Content(role="user", parts=current_parts)
    full_contents = history + [current_content]

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            start_t = time.time()
            response = await ai_client.aio.models.generate_content(
                model=model_name,
                contents=full_contents,
                config=APP_CONFIG,
            )
            if response and response.text:
                elapsed = time.time() - start_t
                print(f"⚡ {model_name} responded in {elapsed:.2f}s")
                reply_text = response.text

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
                    usage_stats["total_tokens"] += tokens

                add_to_history(chat_id, "user", current_parts)
                add_to_history(chat_id, "model", [types.Part.from_text(text=reply_text)])
                return reply_text
        except Exception as e:
            print(f"⚠️ {model_name} failed ({e}), instant failover...")
            last_error = e
            continue

    if last_error:
        raise last_error
    return "Unable to generate a response at this moment."


async def send_clean_message(update: Update, text: str):
    """Deliver clean plain-text messages safely across Telegram character boundaries."""
    max_length = 4000
    if len(text) <= max_length:
        await update.message.reply_text(text, parse_mode=None)
        return

    chunks = []
    current_chunk = ""
    for paragraph in text.split("\n"):
        if len(current_chunk) + len(paragraph) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            while len(paragraph) > max_length:
                chunks.append(paragraph[:max_length])
                paragraph = paragraph[max_length:]
            current_chunk = paragraph
        else:
            current_chunk += ("\n" if current_chunk else "") + paragraph

    if current_chunk:
        chunks.append(current_chunk)

    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=None)


# --- Debug / Activity Logger ---
async def log_incoming_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log every incoming message to terminal so you can see live activity."""
    if update.message and update.effective_user:
        user_name = update.effective_user.first_name
        if update.message.text:
            print(f"\n📩 [Query from {user_name}]: \"{update.message.text}\"")
        elif update.message.photo:
            print(f"\n🖼️ [Photo from {user_name}]")
        elif update.message.document:
            print(f"\n📄 [Document from {user_name}]: {update.message.document.file_name}")


# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_histories.pop(chat_id, None)
    user_name = update.effective_user.first_name if update.effective_user else "there"
    welcome_text = (
        f"Hello {user_name}. I am your AI assistant.\n\n"
        "Capabilities:\n"
        "- Fast, adaptable answers tailored to your exact question\n"
        "- Contextual follow-up discussions\n"
        "- Photo and diagram analysis\n"
        "- PDF and document summaries\n\n"
        "Commands:\n"
        "/new - Clear memory & start a fresh topic\n"
        "/help - Instructions"
    )
    await update.message.reply_text(welcome_text, parse_mode=None)
    print(f"📤 [Sent Welcome to {user_name}]")


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history so the user can switch topics."""
    chat_id = update.effective_chat.id
    chat_histories.pop(chat_id, None)
    await update.message.reply_text("Conversation memory cleared. Ready for a new topic.", parse_mode=None)
    print(f"🧹 [Cleared memory for chat {chat_id}]")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Usage Guide:\n\n"
        "1. Ask short questions for quick answers, or ask for details for deep explanations.\n"
        "2. Ask follow-up questions naturally in the same chat.\n"
        "3. Send any image or PDF document for analysis.\n"
        "4. Send /new whenever you want to clear memory and start fresh."
    )
    await update.message.reply_text(help_text, parse_mode=None)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show usage history and analytics (restricted if ADMIN_USER_ID is configured)."""
    user_id = update.effective_user.id

    if ADMIN_USER_ID and str(user_id) != str(ADMIN_USER_ID).strip():
        await update.message.reply_text("Access denied. This command is restricted to the bot owner.", parse_mode=None)
        return

    elapsed = int(time.time() - BOT_START_TIME)
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    uptime_str = f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m"

    total_requests = usage_stats["text_count"] + usage_stats["photo_count"] + usage_stats["doc_count"]
    recent_list = "\n".join(usage_stats["recent_logs"]) if usage_stats["recent_logs"] else "No activity recorded yet."

    stats_text = (
        "BOT USAGE REPORT\n\n"
        f"Uptime (since last restart): {uptime_str}\n"
        f"Total Unique Users: {len(usage_stats['unique_users'])}\n"
        f"Total Requests Processed: {total_requests}\n"
        f"- Text Questions: {usage_stats['text_count']}\n"
        f"- Photos Reviewed: {usage_stats['photo_count']}\n"
        f"- PDFs / Docs Reviewed: {usage_stats['doc_count']}\n"
        f"Total Gemini Tokens Used: {usage_stats['total_tokens']}\n\n"
        "RECENT ACTIVITY:\n"
        f"{recent_list}"
    )

    if not ADMIN_USER_ID:
        stats_text += (
            f"\n\nPRIVACY TIP:\n"
            f"Your Telegram User ID is {user_id}.\n"
            f"To lock /stats so only you can see it, add ADMIN_USER_ID={user_id} to your Render Environment Variables."
        )

    await update.message.reply_text(stats_text, parse_mode=None)


# --- Message Handlers ---
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name if update.effective_user else "User"

    record_activity(user_id, user_name, "text", user_text)
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        parts = [types.Part.from_text(text=user_text)]
        reply_content = await generate_gemini_response(chat_id, parts)
        await send_clean_message(update, reply_content)
        print(f"📤 [Replied to {user_name}]")
    except Exception as e:
        print(f"❌ [Error]: {e}")
        await update.message.reply_text(f"Error: {e}", parse_mode=None)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    photo = update.message.photo[-1]
    caption = update.message.caption or (
        "Analyze this image clearly in clean plain text. Adapt your depth to the image complexity "
        "without using formatting symbols like # or *."
    )
    user_name = update.effective_user.first_name if update.effective_user else "User"

    record_activity(user_id, user_name, "photo", update.message.caption or "Image uploaded")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    temp_path = None
    try:
        file = await context.bot.get_file(photo.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            temp_path = tf.name
        await file.download_to_drive(custom_path=temp_path)

        with open(temp_path, "rb") as f:
            image_bytes = f.read()

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )
        parts = [image_part, types.Part.from_text(text=caption)]
        reply_content = await generate_gemini_response(chat_id, parts)
        await send_clean_message(update, reply_content)
        print(f"📤 [Replied to photo from {user_name}]")
    except Exception as e:
        print(f"❌ [Error with photo]: {e}")
        await update.message.reply_text(f"Error reviewing image: {e}", parse_mode=None)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    doc = update.message.document
    caption = update.message.caption or (
        "Analyze and summarize this document thoroughly in clean plain text without using symbols like # or *."
    )
    user_name = update.effective_user.first_name if update.effective_user else "User"

    if doc.file_size and doc.file_size > 15 * 1024 * 1024:
        await update.message.reply_text("This file is larger than 15MB. Please send a smaller PDF.", parse_mode=None)
        return

    record_activity(user_id, user_name, "doc", doc.file_name or "PDF document")
    status_msg = await update.message.reply_text("Analyzing your document... Please give me a moment.", parse_mode=None)
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    temp_path = None
    uploaded_file = None
    try:
        file = await context.bot.get_file(doc.file_id)
        file_ext = os.path.splitext(doc.file_name or "")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tf:
            temp_path = tf.name

        await file.download_to_drive(custom_path=temp_path)
        uploaded_file = await ai_client.aio.files.upload(file=temp_path)

        parts = [uploaded_file, types.Part.from_text(text=caption)]
        reply_content = await generate_gemini_response(chat_id, parts)
        await send_clean_message(update, reply_content)
        print(f"📤 [Replied to document from {user_name}]")
    except Exception as e:
        print(f"❌ [Error with document]: {e}")
        await update.message.reply_text(f"Error reviewing document: {e}", parse_mode=None)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        if uploaded_file and hasattr(uploaded_file, "name"):
            try:
                await ai_client.aio.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                pass


async def post_init(application):
    """Executed automatically once the bot connects to Telegram."""
    me = await application.bot.get_me()
    print("\n" + "=" * 55)
    print("AI ASSISTANT BOT IS ONLINE!")
    print(f"Bot Name:     {me.first_name}")
    print(f"Username:     @{me.username}")
    print(f"Direct Link:   https://t.me/{me.username}")
    print("=" * 55)
    print("Listening for messages...\n")


def main():
    is_cloud = start_cloud_health_server()
    if is_cloud:
        # Briefly wait for old Render container to cleanly disconnect during deploys
        print("⏳ Waiting 5s for smooth cloud deployment handoff...")
        time.sleep(5)

    print("Connecting to Telegram...")
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(TypeHandler(Update, log_incoming_update), group=-1)
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("new", new_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
