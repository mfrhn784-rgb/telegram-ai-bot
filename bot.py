import os
import io
import logging
import threading
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

# 3. Optional Cloud Health Server (Allows free deployment on Render / Koyeb)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"AI Telegram Bot is active!")

    def log_message(self, format, *args):
        pass  # Silence web server access logs

def start_cloud_health_server():
    port = int(os.getenv("PORT", 0))
    if port:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"🌐 Cloud health check listening on port {port}")

# 4. Initialize Gemini AI Client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Ultra-compact, token-efficient system instruction
CLEAN_SYSTEM_INSTRUCTION = (
    "You are a friendly, fast, and smart AI assistant.\n"
    "Rules:\n"
    "1. NO SYMBOLS: Never use #, *, **, $, _, ~, `, or {}. Write in clean plain text only.\n"
    "2. TOKEN SAVER: Be extremely brief, direct, and compact. 2 to 4 sentences or clean lines. No filler.\n"
    "3. ACCURACY: Give up-to-date and accurate facts on any topic."
)

APP_CONFIG = types.GenerateContentConfig(
    system_instruction=CLEAN_SYSTEM_INSTRUCTION,
    max_output_tokens=400,
    temperature=0.5,
)

MODELS_TO_TRY = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]


async def generate_gemini_response(contents):
    """Generate content asynchronously with automatic fallback and minimal token usage."""
    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            print(f"🤖 Processing with {model_name}...")
            response = await ai_client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=APP_CONFIG,
            )
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"⚠️ {model_name} issue: {e}. Trying fallback...")
            last_error = e
            continue

    if last_error:
        raise last_error
    return "I couldn't generate a response. Please try asking again."


async def send_clean_message(update: Update, text: str):
    """Deliver clean plain-text messages safely."""
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
    user_name = update.effective_user.first_name if update.effective_user else "there"
    welcome_text = (
        f"Hello {user_name}! I am your fast AI assistant.\n\n"
        "What I can do for you:\n"
        "- Compact, direct answers to any question\n"
        "- Studies, general knowledge, and brainstorming\n"
        "- Quick image and photo review\n"
        "- PDF summaries\n\n"
        "Send any message or file to get started!"
    )
    await update.message.reply_text(welcome_text, parse_mode=None)
    print(f"📤 [Sent Welcome to {user_name}]")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "How to use me:\n\n"
        "1. Send any question to get a fast, compact answer.\n"
        "2. Send any photo to analyze it.\n"
        "3. Send any PDF to get a summary.\n\n"
        "Commands:\n"
        "/start - Restart the bot\n"
        "/help - Instructions"
    )
    await update.message.reply_text(help_text, parse_mode=None)


# --- Message Handlers ---
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    chat_id = update.effective_chat.id
    user_name = update.effective_user.first_name if update.effective_user else "User"

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        reply_content = await generate_gemini_response(user_text)
        await send_clean_message(update, reply_content)
        print(f"📤 [Replied to {user_name}]")
    except Exception as e:
        print(f"❌ [Error]: {e}")
        await update.message.reply_text(f"Error: {e}", parse_mode=None)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    chat_id = update.effective_chat.id
    photo = update.message.photo[-1]
    caption = update.message.caption or "Analyze this image briefly in plain text without formatting symbols."
    user_name = update.effective_user.first_name if update.effective_user else "User"

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        file = await context.bot.get_file(photo.file_id)
        image_stream = io.BytesIO()
        await file.download_to_memory(out=image_stream)
        image_bytes = image_stream.getvalue()

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )
        reply_content = await generate_gemini_response([image_part, caption])
        await send_clean_message(update, reply_content)
        print(f"📤 [Replied to photo from {user_name}]")
    except Exception as e:
        print(f"❌ [Error with photo]: {e}")
        await update.message.reply_text(f"Error reviewing image: {e}", parse_mode=None)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return

    chat_id = update.effective_chat.id
    doc = update.message.document
    mime_type = doc.mime_type or "application/pdf"
    caption = update.message.caption or "Summarize this document briefly in plain text without formatting symbols."
    user_name = update.effective_user.first_name if update.effective_user else "User"

    if doc.file_size and doc.file_size > 20 * 1024 * 1024:
        await update.message.reply_text("The file is larger than 20MB (Telegram limit).", parse_mode=None)
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        file = await context.bot.get_file(doc.file_id)
        doc_stream = io.BytesIO()
        await file.download_to_memory(out=doc_stream)
        doc_bytes = doc_stream.getvalue()

        doc_part = types.Part.from_bytes(
            data=doc_bytes,
            mime_type=mime_type,
        )
        reply_content = await generate_gemini_response([doc_part, caption])
        await send_clean_message(update, reply_content)
        print(f"📤 [Replied to document from {user_name}]")
    except Exception as e:
        print(f"❌ [Error with document]: {e}")
        await update.message.reply_text(f"Error reviewing document: {e}", parse_mode=None)


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
    # Start optional cloud health server if hosted on Render / Koyeb
    start_cloud_health_server()

    print("Connecting to Telegram...")
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(TypeHandler(Update, log_incoming_update), group=-1)
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    app.run_polling()


if __name__ == "__main__":
    main()
