"""
utils.py — v2: роутинг Haiku/Sonnet, prompt caching, кредитная система, троттлинг безлимита
"""
import asyncio
import io
import re
import math
import time
import urllib.parse
import logging
from typing import Optional, Tuple
from collections import defaultdict

import aiohttp
import edge_tts
from aiogram.types import Message, BufferedInputFile
from aiogram.enums import ParseMode
from bs4 import BeautifulSoup

import config
import database as db

log = logging.getLogger("broks_ai")

# ══════════════════════════════════════════════════════
#  МОДЕЛИ
# ══════════════════════════════════════════════════════
SONNET_MODEL = "claude-sonnet-4-20250514"
HAIKU_MODEL = "claude-haiku-4-5-20241022"

# ══════════════════════════════════════════════════════
#  RATE LIMITER — 10 req/min per user
# ══════════════════════════════════════════════════════
_rate_limiter: dict[int, list[float]] = defaultdict(list)
RATE_LIMIT = 10
RATE_WINDOW = 60  # секунд

def check_rate_limit(user_id: int) -> bool:
    """Проверяет rate limit. True = можно, False = лимит."""
    now = time.time()
    _rate_limiter[user_id] = [t for t in _rate_limiter[user_id] if now - t < RATE_WINDOW]
    if len(_rate_limiter[user_id]) >= RATE_LIMIT:
        return False
    _rate_limiter[user_id].append(now)
    return True

# ══════════════════════════════════════════════════════
#  КЛАССИФИКАЦИЯ ЗАПРОСА → роутинг модели + max_tokens
# ══════════════════════════════════════════════════════

# Паттерны для простых запросов (→ Haiku)
_SIMPLE_PATTERNS = [
    r'^(привет|здравствуй|хай|hi|hello|hey)',
    r'^(спасибо|благодарю|thanks|thank you)',
    r'^(да|нет|ок|ok|ладно|хорошо|понял)',
    r'^(что такое|кто такой|когда |где |сколько )',
    r'^(переведи|translate)',
    r'(как дела|как ты|что нового)',
    r'^[^а-яa-z]*$',  # только эмодзи/знаки
]

# Паттерны для сложных запросов (→ Sonnet + больше токенов)
_COMPLEX_PATTERNS = [
    r'(напиши код|write code|программ|скрипт|функци|алгоритм)',
    r'(анализ|проанализируй|analyze|разбери|оцени)',
    r'(создай|разработай|спроектируй|build|create|develop)',
    r'(объясни подробно|explain in detail|пошагово|step by step)',
    r'(статья|эссе|сочинение|текст на \d+ |essay|article)',
    r'(договор|контракт|документ|contract)',
    r'(бизнес.план|маркетинг|стратегия)',
    r'(сравни|compare|отличия|разница между)',
    r'(реши задачу|math|формула|уравнение)',
]

def classify_request(text: str) -> Tuple[str, int]:
    """
    Классифицирует запрос.
    Returns: (model, max_tokens)
      - 'haiku', 128   — простые (приветствия, да/нет)
      - 'haiku', 384   — средние (факты, переводы)
      - 'sonnet', 512  — обычные
      - 'sonnet', 1024 — сложные (код, анализ, документы)
    """
    text_lower = text.lower().strip()
    
    # Очень короткие сообщения → Haiku, мало токенов
    if len(text_lower) < 15:
        for p in _SIMPLE_PATTERNS:
            if re.search(p, text_lower):
                return HAIKU_MODEL, 128
        return HAIKU_MODEL, 384
    
    # Проверяем на сложность
    for p in _COMPLEX_PATTERNS:
        if re.search(p, text_lower):
            return SONNET_MODEL, 1024
    
    # Длинный текст (>200 символов) скорее сложный
    if len(text_lower) > 200:
        return SONNET_MODEL, 512
    
    # По умолчанию — Haiku для коротких, Sonnet для средних
    if len(text_lower) < 60:
        return HAIKU_MODEL, 384
    
    return SONNET_MODEL, 512


# ══════════════════════════════════════════════════════
#  ПОДСЧЁТ ТОКЕНОВ (приблизительный)
# ══════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов: ~4 символа = 1 токен для русского, ~3.5 для английского"""
    if not text:
        return 0
    # Русский текст: ~3.5 символа на токен
    return max(1, len(text) // 4)


def calc_conditional_tokens(input_tokens: int, output_tokens: int) -> float:
    """
    Условные токены: input×0.2 + output×1.0
    Коэффициент 0.2 отражает разницу цен API: input $3/1M vs output $15/1M
    """
    return input_tokens * 0.2 + output_tokens * 1.0


# ══════════════════════════════════════════════════════
#  ОБРЕЗКА ИСТОРИИ — максимум 4000 токенов
# ══════════════════════════════════════════════════════

def trim_history(messages: list[dict], max_tokens: int = 4000) -> list[dict]:
    """
    Обрезает историю до max_tokens.
    Оставляет последние 3 пары сообщений.
    """
    if not messages:
        return []
    
    # Считаем токены с конца
    total = 0
    keep = []
    for msg in reversed(messages):
        msg_tokens = estimate_tokens(msg.get("content", ""))
        if total + msg_tokens > max_tokens and len(keep) >= 6:  # минимум 3 пары
            break
        total += msg_tokens
        keep.insert(0, msg)
    
    return keep


# ══════════════════════════════════════════════════════
#  ANTHROPIC CLAUDE API — с роутингом и кэшированием
# ══════════════════════════════════════════════════════

async def chat_with_claude(user_id: int, text: str, force_haiku: bool = False) -> dict:
    """
    Отправляет запрос в Claude API с роутингом модели.
    
    Returns: dict {
        'text': str,           # ответ
        'input_tokens': int,   # токены на входе
        'output_tokens': int,  # токены на выходе
        'model': str,          # какую модель использовали
        'cached': bool,        # был ли кэш
    }
    """
    # Классификация запроса
    model, max_tokens = classify_request(text)
    
    # Принудительно Haiku (для троттлинга безлимита)
    if force_haiku:
        model = HAIKU_MODEL
        max_tokens = min(max_tokens, 512)
    
    # Получаем историю и обрезаем
    history = await db.get_history(user_id)
    trimmed = trim_history(history, max_tokens=4000)
    
    messages = []
    for msg in trimmed:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": text})
    
    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    
    # Prompt Caching — системный промпт в кэш
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": config.SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": messages,
    }
    
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json()
            
            if "content" in data and len(data["content"]) > 0:
                content = data["content"][0].get("text", "")
                if content and content.strip():
                    # Получаем реальное использование токенов
                    usage = data.get("usage", {})
                    input_tok = usage.get("input_tokens", estimate_tokens(text))
                    output_tok = usage.get("output_tokens", estimate_tokens(content))
                    cache_read = usage.get("cache_read_input_tokens", 0)
                    
                    # Сохраняем в историю
                    await db.add_message(user_id, "user", text)
                    await db.add_message(user_id, "assistant", content)
                    
                    return {
                        'text': content,
                        'input_tokens': input_tok,
                        'output_tokens': output_tok,
                        'model': model,
                        'cached': cache_read > 0,
                    }
            
            if "error" in data:
                log.error(f"[Claude API] Error: {data['error']}")
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return {
                    'text': f"Ошибка API: {data['error'].get('message', 'Unknown')}",
                    'input_tokens': 0, 'output_tokens': 0,
                    'model': model, 'cached': False,
                }
        
        except Exception as e:
            log.error(f"[Claude API] Exception: {e}")
            if attempt < 2:
                await asyncio.sleep(1)
                continue
    
    return {
        'text': "Не удалось получить ответ. Попробуй ещё раз.",
        'input_tokens': 0, 'output_tokens': 0,
        'model': HAIKU_MODEL, 'cached': False,
    }


# ══════════════════════════════════════════════════════
#  КРЕДИТНАЯ СИСТЕМА — подсчёт и списание
# ══════════════════════════════════════════════════════

async def process_credits(user_id: int, input_tokens: int, output_tokens: int) -> dict:
    """
    Считает условные токены, обновляет накопитель, списывает кредиты.
    Returns: dict с инфо о списании
    """
    conditional = calc_conditional_tokens(input_tokens, output_tokens)
    
    # Получаем текущий накопитель из БД
    import aiosqlite
    async with aiosqlite.connect(config.DB_PATH) as conn:
        row = await conn.execute(
            "SELECT token_accumulator, bonus_requests FROM users WHERE user_id=?", (user_id,)
        )
        row = await row.fetchone()
        if not row:
            return {'credits_used': 0, 'conditional': conditional}
        
        accumulator = (row[0] or 0) + conditional
        credits_to_deduct = int(accumulator // 1000)
        new_accumulator = accumulator % 1000
        
        if credits_to_deduct > 0:
            await conn.execute(
                "UPDATE users SET token_accumulator=?, bonus_requests=MAX(0, COALESCE(bonus_requests,0)-?) WHERE user_id=?",
                (new_accumulator, credits_to_deduct, user_id)
            )
        else:
            await conn.execute(
                "UPDATE users SET token_accumulator=? WHERE user_id=?",
                (new_accumulator, user_id)
            )
        await conn.commit()
    
    return {
        'credits_used': credits_to_deduct,
        'conditional_tokens': conditional,
        'accumulator': new_accumulator,
    }


# ══════════════════════════════════════════════════════
#  БЕЗЛИМИТ ТРОТТЛИНГ
# ══════════════════════════════════════════════════════

UNLIMITED_THROTTLE_THRESHOLD = 80  # запросов в день до троттлинга

async def should_throttle(user_id: int) -> bool:
    """Проверяет нужно ли троттлить безлимит юзера (>80 запросов/день)"""
    usage = await db.get_usage(user_id)
    return usage.get('ai_requests', 0) >= UNLIMITED_THROTTLE_THRESHOLD


THROTTLE_MESSAGE = (
    "⚙️ Для стабильной работы сервиса качество ответов временно снижено. "
    "Полная мощность восстановится в 00:00."
)


# ══════════════════════════════════════════════════════
#  СТАРЫЕ ФУНКЦИИ — сохраняем совместимость
# ══════════════════════════════════════════════════════

# OpenRouter API (для философского режима и др.)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _get_headers():
    return {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

OPENROUTER_HEADERS = _get_headers()

def clean_response(text: str) -> str:
    """Убирает think-теги Qwen, чистит текст"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    if not text:
        text = "Не удалось получить ответ. Попробуй ещё раз."
    return text[:4000]


async def chat_with_history(user_id: int, user_message: str) -> str:
    """Диалог с памятью — история из БД (OpenRouter)"""
    history = await db.get_history(user_id)
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        payload = {"model": config.GROQ_MODEL, "messages": messages,
                   "max_tokens": config.GROQ_MAX_TOKENS, "temperature": 0.7, "reasoning": {"exclude": True}}
        data = {}
        for attempt in range(3):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    OPENROUTER_URL, headers=_get_headers(), json=payload
                ) as resp:
                    data = await resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                break
            await asyncio.sleep(1)
        reply = clean_response((data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""))
        await db.add_message(user_id, "user", user_message)
        await db.add_message(user_id, "assistant", reply)
        return reply
    except Exception as e:
        return f"❌ Ошибка API: {e}"


async def groq_single(prompt: str, system: str = "", temperature: float = 0.8) -> str:
    """Одиночный запрос без истории (OpenRouter)"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENROUTER_URL, headers=_get_headers(),
                json={"model": config.GROQ_MODEL, "messages": messages,
                      "max_tokens": config.GROQ_MAX_TOKENS, "temperature": temperature, "reasoning": {"exclude": True}}
            ) as resp:
                data = await resp.json()
        return clean_response((data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""))
    except Exception as e:
        return f"❌ Ошибка API: {e}"


# ══════════════════════════════════════════════════════
#  Whisper — голос в текст
# ══════════════════════════════════════════════════════

async def transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    import tempfile, os
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        import whisper
        loop = asyncio.get_event_loop()
        def _transcribe():
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path, language="ru")
            return result["text"].strip()
        text = await loop.run_in_executor(None, _transcribe)
        os.unlink(tmp_path)
        return text if text else "❌ Не удалось распознать речь"
    except Exception as e:
        return f"❌ Ошибка транскрибации: {e}"


# ══════════════════════════════════════════════════════
#  POLLINATIONS.AI — генерация изображений
# ══════════════════════════════════════════════════════

async def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[bytes]:
    enhanced = f"{prompt}, high quality, detailed, professional, 4k, sharp focus, beautiful composition"
    encoded = urllib.parse.quote(enhanced)
    url = f"{config.POLLINATIONS_BASE}/{encoded}?width={width}&height={height}&seed={hash(prompt) % 9999}&nologo=true"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    return await resp.read()
    except Exception:
        pass
    return None

async def generate_image_with_retry(prompt: str, attempts: int = 3) -> Optional[bytes]:
    for i in range(attempts):
        result = await generate_image(prompt)
        if result and len(result) > 10_000:
            return result
        await asyncio.sleep(2)
    return None


# ══════════════════════════════════════════════════════
#  MEMEGEN — мемы
# ══════════════════════════════════════════════════════

async def generate_meme(template: str, top_text: str, bottom_text: str) -> str:
    top = urllib.parse.quote(top_text.replace(" ", "_") or "_")
    bottom = urllib.parse.quote(bottom_text.replace(" ", "_") or "_")
    return f"{config.MEMEGEN_API}/images/{template}/{top}/{bottom}.jpg"

async def get_popular_templates() -> list[dict]:
    return [
        {"id": "drake", "name": "Drake (не это / это)"},
        {"id": "doge", "name": "Doge"},
        {"id": "distracted", "name": "Отвлечённый парень"},
        {"id": "success", "name": "Success Kid"},
        {"id": "fry", "name": "Futurama Fry"},
        {"id": "woman-yelling-at-cat", "name": "Женщина кричит на кота"},
        {"id": "two-buttons", "name": "Две кнопки"},
        {"id": "this-is-fine", "name": "This is fine"},
        {"id": "change-my-mind", "name": "Change My Mind"},
        {"id": "gru-plan", "name": "План Гру"},
    ]


# ══════════════════════════════════════════════════════
#  MYMEMORY — перевод
# ══════════════════════════════════════════════════════

async def translate_text(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
    if source_lang == "auto":
        lang_pair = f"ru|{target_lang}"
    else:
        lang_pair = f"{source_lang}|{target_lang}"
    params = {"q": text, "langpair": lang_pair}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.MYMEMORY_API, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
                translated = data.get("responseData", {}).get("translatedText", "")
                if translated and translated.lower() != "mymemory warning:":
                    return translated
                return "❌ Не удалось перевести текст."
    except Exception as e:
        return f"❌ Ошибка перевода: {e}"


# ══════════════════════════════════════════════════════
#  WEB PARSER
# ══════════════════════════════════════════════════════

async def parse_website(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20), allow_redirects=True) as resp:
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status}"}
                html = await resp.text(errors="ignore")
    except Exception as e:
        return {"error": f"Не удалось загрузить: {e}"}
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title else "Без заголовка"
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    raw_text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in raw_text.splitlines() if len(l.strip()) > 30]
    clean_text = "\n".join(lines[:150])
    text_preview = clean_text[:500] + "..." if len(clean_text) > 500 else clean_text
    summary = await groq_single(
        f"Сделай краткое резюме (5-7 предложений) текста со страницы '{url}':\n\n{clean_text[:3000]}",
        system="Ты помощник-аналитик. Делаешь краткие саммари на русском.", temperature=0.3,
    )
    return {"url": url, "title": title, "text_preview": text_preview, "summary": summary}


# ══════════════════════════════════════════════════════
#  БИЗНЕС PROMPTS
# ══════════════════════════════════════════════════════

async def analyze_business_idea(idea: str) -> str:
    system = "Ты опытный бизнес-аналитик и стратегический консультант. Отвечаешь структурированно на русском."
    prompt = f"""Проанализируй бизнес-идею: {idea}

🎯 **ОЦЕНКА** (от 1 до 10)
👥 **ЦЕЛЕВАЯ АУДИТОРИЯ**
⚠️ **РИСКИ**
💰 **МОНЕТИЗАЦИЯ**
🚀 **ПЛАН ЗАПУСКА (MVP)**
💡 **ИТОГ**"""
    return await groq_single(prompt, system=system, temperature=0.4)


async def generate_contract(details: str) -> str:
    system = "Ты юрист по российскому гражданскому праву. Составляешь договоры по ГК РФ."
    prompt = f"Составь договор: {details}"
    return await groq_single(prompt, system=system, temperature=0.2)


async def generate_social_content(topic: str, platform: str) -> str:
    platform_guides = {
        "instagram": "150-300 символов + хэштеги.",
        "telegram": "500-1500 символов. Структура с заголовком.",
        "vk": "200-500 символов, 3-5 хэштегов.",
        "twitter_x": "До 280 символов.",
        "tiktok": "100-150 символов + хэштеги.",
    }
    guide = platform_guides.get(platform, "200-500 символов.")
    return await groq_single(
        f"Создай 3 варианта поста для {platform.upper()} на тему: \"{topic}\"\n{guide}",
        system="Ты SMM-специалист.", temperature=0.9,
    )


# ══════════════════════════════════════════════════════
#  EDGE-TTS
# ══════════════════════════════════════════════════════

TTS_VOICES = {
    "ru-RU-DmitryNeural":   "🎙 Дмитрий (муж.)",
    "ru-RU-SvetlanaNeural": "🎙 Светлана (жен.)",
    "ru-RU-DariyaNeural":   "🎙 Дарья (жен., мягкий)",
    "en-US-AndrewNeural":   "🎙 Andrew (eng, муж.)",
    "en-US-JennyNeural":    "🎙 Jenny (eng, жен.)",
}

async def text_to_speech(text: str, voice: str = "ru-RU-DmitryNeural") -> Optional[bytes]:
    clean = re.sub(r"[*_`#\[\]()~>+=|{}.!\\]", "", text)
    clean = re.sub(r"\n+", ". ", clean).strip()
    if len(clean) > 3000:
        clean = clean[:3000] + "..."
    try:
        communicate = edge_tts.Communicate(clean, voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        data = buf.read()
        return data if len(data) > 1000 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════
#  SMART REPLY
# ══════════════════════════════════════════════════════

async def safe_answer(message, text: str, reply_markup=None):
    try:
        await message.answer(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    except Exception:
        await message.answer(text, reply_markup=reply_markup)

async def smart_reply(message: Message, text: str, reply_markup=None, force_text: bool = False) -> None:
    user_id = message.from_user.id
    mode = await db.get_response_mode(user_id)
    if force_text or mode == "text":
        if len(text) > 4000:
            for i in range(0, len(text), 4000):
                chunk = text[i:i+4000]
                is_last = (i + 4000 >= len(text))
                await safe_answer(message, chunk, reply_markup=reply_markup if is_last else None)
        else:
            await safe_answer(message, text, reply_markup=reply_markup)
        return
    voice_name = await db.get_tts_voice(user_id)
    audio_bytes = await text_to_speech(text, voice=voice_name)
    if audio_bytes:
        try:
            await message.answer_voice(voice=BufferedInputFile(audio_bytes, filename="reply.mp3"), reply_markup=reply_markup)
        except Exception:
            await safe_answer(message, text, reply_markup=reply_markup)
    else:
        await safe_answer(message, text, reply_markup=reply_markup)


# ══════════════════════════════════════════════════════
#  ФИЛОСОФСКИЙ РЕЖИМ
# ══════════════════════════════════════════════════════

PHILOSOPHICAL_SYSTEM_PROMPT = (
    "Ты — дерзкий мудрец, который курит и говорит как есть, без цензуры. "
    "Стиль: уличная философия + нуар. Каждый ответ начинай с ремарки: "
    "(_затягивается_, _выпускает дым_). Не более 200 слов."
)

def _generate_drag_wav(duration: float = 2.2, rate: int = 22050) -> bytes:
    import random as _rng, struct, wave
    rng = _rng.Random(1337)
    n = int(duration * rate)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        frames = []
        for i in range(n):
            t = i / rate
            env = (t / 0.08) ** 2 if t < 0.08 else (1.0 if t < 0.45 else math.exp(-(t - 0.45) * 1.7))
            noise = rng.gauss(0, 1)
            crackle = rng.gauss(0, 3.8) if rng.random() < 0.014 else 0.0
            breath = math.sin(2 * math.pi * 88 * t) * math.sin(2 * math.pi * 1.7 * t)
            raw = (noise * 0.44 + crackle * 0.26 + breath * 0.17) * env * 0.54
            raw = max(-1.0, min(1.0, raw))
            frames.append(struct.pack('<h', int(raw * 30000)))
        wf.writeframes(b''.join(frames))
    return buf.getvalue()

_DRAG_WAV_CACHE: Optional[bytes] = None

def get_drag_sound() -> bytes:
    global _DRAG_WAV_CACHE
    if _DRAG_WAV_CACHE is None:
        _DRAG_WAV_CACHE = _generate_drag_wav()
    return _DRAG_WAV_CACHE

async def philosophical_reply(message: Message, user_text: str, reply_markup=None) -> None:
    thinking_msg = await message.answer("💭 Думаю...")
    user_id = message.from_user.id
    history = await db.get_history(user_id)
    messages_payload = [{"role": "system", "content": PHILOSOPHICAL_SYSTEM_PROMPT}]
    messages_payload.extend(history)
    messages_payload.append({"role": "user", "content": user_text})
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENROUTER_URL, headers=OPENROUTER_HEADERS,
                json={"model": config.GROQ_MODEL, "messages": messages_payload,
                      "max_tokens": 400, "temperature": 0.95, "reasoning": {"exclude": True}}
            ) as resp:
                data = await resp.json()
        reply = clean_response((data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""))
    except Exception as e:
        reply = f"❌ {e}"
    await db.add_message(user_id, "user", user_text)
    await db.add_message(user_id, "assistant", reply)
    try:
        await thinking_msg.delete()
    except Exception:
        pass
    await smart_reply(message, reply)
