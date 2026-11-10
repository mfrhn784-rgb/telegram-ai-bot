import os
import io
import tempfile
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

# 3. Cloud Health Server (Keeps Render / Koyeb alive without OOM)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"High-IQ AI Telegram Bot is active!")

    def log_message(self, format, *args):
        pass

def start_cloud_health_server():
    port = int(os.getenv("PORT", 0))
    if port:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"🌐 Cloud health check listening on port {port}")

# 4. Initialize Gemini AI Client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# High-IQ, comprehensive, clean plain-text instruction
HIGH_IQ_SYSTEM_INSTRUCTION = (
    "You are an exceptionally intelligent, articulate, and deeply knowledgeable AI mind.\n"
    "Your objective is to provide comprehensive, thorough, and crystal-clear explanations with high intellectual depth.\n\n"
    "CORE PRINCIPLES:\n"
    "1. THOROUGH & WELL-DEFINED: Never give short or dismissive answers. Fully define the subject, break down how it works, "
    "explain background context, nuances, and real-world applications. Ensure the user walks away with complete mastery of the topic.\n"
    "2. NO MEMORY / NO FOLLOW-UPS: Treat every prompt as an independent, standalone request. Never assume or refer to old conversations. "
    "Crucially: NEVER add conversational follow-up questions at the end (e.g., do NOT write 'Do you want more details?', 'Let me know if you need help', etc.). End cleanly once the explanation is complete.\n"
    "3. CLEAN PLAIN TEXT (ZERO SYMBOLS): Telegram does not render markdown well. Never use symbols like #, ##, *, **, $, _, ~, `, or {}.\n"
    "To organize your response cleanly, use neat line breaks and readable plain capital labels (such as OVERVIEW, CORE DEFINITION, HOW IT WORKS, KEY TAKEAWAYS), or standard numbers (1, 2, 3).\n"
    "4. SCIENCE, FORMULAS & REACTIONS: When explaining reactions or formulas, write them out clearly in words first, followed by clean standard keyboard text without confusing code symbols.\n"
    "5. IMAGES & PDFS: Provide deep, systematic, and well-structured analysis covering all key elements in detail."
)

APP_CONFIG = types.GenerateContentConfig(
    system_instruction=HIGH_IQ_SYSTEM_INSTRUCTION,
    max_output_tokens=2048,  # Generous headroom for deep, thorough answers
    temperature=0.6,
)

# Priority list of models (prioritizes 3.8-flash for peak reasoning and high IQ)
MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]


async def generate_gemini_response(contents):
    """Generate content asynchronously with automatic fallback and high intellectual depth."""
    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            print(f"🧠 High-IQ reasoning with {model_name}...")
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
    return "Unable to generate a response at this moment."


async def send_clean_message(update: Update, text: str):
    """Deliver clean plain-text messages safely across Telegram character boundaries."""
    max_length = 4000
    if len(text) <= max_length:
        await update.message.reply_text(text, parse_mode=None)
        return

    # Clean chunking for long, in-depth explanations
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
        f"Hello {user_name}. I am your advanced AI assistant.\n\n"
        "Capabilities:\n"
        "- Comprehensive, deeply articulated answers across all domains\n"
        "- Advanced scientific, mathematical, and conceptual explanations\n"
        "- High-detail photo and diagram analysis\n"
        "- Exhaustive PDF and document reviews\n\n"
        "Send any inquiry, image, or document to begin."
    )
    await update.message.reply_text(welcome_text, parse_mode=None)
    print(f"📤 [Sent Welcome to {user_name}]")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Usage Guide:\n\n"
        "1. Send any query to receive a thorough, comprehensive explanation.\n"
        "2. Send an image to receive a detailed breakdown of its contents.\n"
        "3. Send a PDF document to receive an in-depth analytical summary.\n\n"
        "Commands:\n"
        "/start - Initialize bot\n"
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
        print(f"📤 [Replied in detail to {user_name}]")
    except Exception as e:
        print(f"❌ [Error]: {e}")
        await update.message.reply_text(f"Error: {e}", parse_mode=None)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    chat_id = update.effective_chat.id
    photo = update.message.photo[-1]
    caption = update.message.caption or (
        "Please provide a comprehensive, deeply analytical breakdown of this image in clean plain text. "
        "Explain all concepts, objects, text, and implications thoroughly without using formatting symbols like # or *."
    )
    user_name = update.effective_user.first_name if update.effective_user else "User"

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
        reply_content = await generate_gemini_response([image_part, caption])
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
    doc = update.message.document
    caption = update.message.caption or (
        "Please provide an exhaustive, deeply structured analytical review of this document in clean plain text. "
        "Define all critical concepts, explain background logic, highlight key takeaways, and ensure thorough depth "
        "without using symbols like # or *."
    )
    user_name = update.effective_user.first_name if update.effective_user else "User"

    if doc.file_size and doc.file_size > 15 * 1024 * 1024:
        await update.message.reply_text("This file is larger than 15MB. Please send a smaller PDF.", parse_mode=None)
        return

    status_msg = await update.message.reply_text("Analyzing your document in depth... Please give me a moment.", parse_mode=None)
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

        reply_content = await generate_gemini_response([uploaded_file, caption])
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
    print("HIGH-IQ AI BOT IS ONLINE!")
    print(f"Bot Name:     {me.first_name}")
    print(f"Username:     @{me.username}")
    print(f"Direct Link:   https://t.me/{me.username}")
    print("=" * 55)
    print("Ready for complex inquiries...\n")


def main():
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
