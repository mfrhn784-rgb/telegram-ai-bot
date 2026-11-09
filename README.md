# Telegram AI Assistant Bot (Text, Images, PDFs)

A multimodal AI bot for Telegram powered by Google Gemini and `python-telegram-bot`.

## Features
- 💬 **Text Chat:** Answer questions, code, write, brainstorm.
- 🖼️ **Image Review:** Send any photo to analyze, describe, or extract text from it.
- 📄 **PDF Review:** Send any PDF document to summarize and extract insights.

## Quick Setup

1. **Add your keys to `.env`:**
   Open [.env](file:///C:/Users/HAFNA/.gemini/antigravity/scratch/telegram_bot/.env) and insert:
   - `TELEGRAM_BOT_TOKEN`: The token you received from `@BotFather`.
   - `GEMINI_API_KEY`: Free key from [Google AI Studio](https://aistudio.google.com/).

2. **Run the bot:**
   ```powershell
   cd C:\Users\HAFNA\.gemini\antigravity\scratch\telegram_bot
   python bot.py
   ```

3. **Test in Telegram:**
   - Search for your bot handle on Telegram.
   - Click **Start** or send `/start`.
   - Send text, a photo, or a PDF.
