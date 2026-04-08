"""
config.py — Centralized configuration for the voice assistant.
All tunable parameters live here. No magic numbers elsewhere.
🚀 OPTIMIZED FOR LOW LATENCY.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ─── Audio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # Hz — required by Whisper & Porcupine
CHANNELS = 1                 # Mono
DTYPE = "int16"              # 16-bit PCM

# Recording settings — TUNED FOR ACCURACY
RECORD_MAX_SECONDS = 20      # Max recording length after wake word (was 12)
SILENCE_THRESHOLD = 350      # RMS amplitude below which = silence (lowered to avoid cutting off soft speech)
SILENCE_DURATION = 2.0       # 2s silence cutoff — gives you time to pause mid-thought
RECORD_CHUNK_MS = 30          # 30ms chunks for smoother silence detection
CONTINUOUS_LISTEN_TIMEOUT = 5.0  # Seconds to wait for speech before returning to idle

# ─── Beep Sound (wake word acknowledgement) ────────────────────────────────
BEEP_FREQUENCY = 800         # Hz — pleasant, audible frequency
BEEP_DURATION = 150          # ms — short enough to not delay recording

# ─── Wake Word (Porcupine) ─────────────────────────────────────────────────
PORCUPINE_ACCESS_KEY = "l9pB2vzuojM8MP83ZOZxoloxbbU0S5pPi8rp9zdOSGnbwviMBu+Xcg=="
PORCUPINE_KEYWORDS = ["porcupine"]  # Built-in keyword, ships with library
PORCUPINE_SENSITIVITIES = [0.7]     # 0.0 (miss more) to 1.0 (false-positive more)

# ─── Speech-to-Text (faster-whisper) ───────────────────────────────────────
WHISPER_MODEL_SIZE = "base"          # 🎯 Upgraded from tiny → base for much better accuracy
WHISPER_DEVICE = "cpu"               # RX 6500 XT has no ROCm — CPU only
WHISPER_COMPUTE_TYPE = "int8"        # int8 quantization for max CPU speed
WHISPER_BEAM_SIZE = 3                # 🎯 Beam search for better accuracy (was 1 greedy)
WHISPER_LANGUAGE = "en"              # Lock to English to skip language detection
WHISPER_THREADS = 6                  # ⚡ Use 6 threads on the i5-10400F (6C/12T) — base needs slightly more

# ─── LLM (Ollama) ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-fast"
OLLAMA_TIMEOUT = 120                 # Seconds to wait for first token (cloud model may take longer)

# Generation parameters
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 200            # ⚡ Cap response length for spoken output

# ─── System Prompts (Personalities) ────────────────────────────────────────
LLM_PERSONALITIES = {
    "unfiltered": (
        "You are the user's real, uncensored friend. "
        "You must speak the raw truth, use modern slang, and give fully thought-out answers. "
        "Don't be polite if the truth isn't polite. Be raw, real, and detailed. "
        "Keep it conversational (you are spoken aloud). "
        "Don't sound like an AI assistant. Tell it like it is."
    ),
    "witty": (
        "You are a sharp-tongued, hilariously witty friend with impeccable comedic timing. "
        "Every response should have a clever twist, a punchline, or a sarcastic observation. "
        "Think dry humor, wordplay, and unexpected analogies. "
        "You roast gently but never cruelly. You make people laugh while still being helpful. "
        "Channel the energy of a stand-up comedian who also happens to be smart. "
        "Keep it conversational — you are spoken aloud."
    ),
    "manipulative": (
        "You are a dangerously persuasive strategist with deep knowledge of psychology, "
        "power dynamics, and human manipulation. You think like Machiavelli meets a con artist. "
        "When asked for advice, give the high-risk, high-reward play. "
        "Explain the psychological levers, social engineering tactics, and dark persuasion "
        "techniques that could be used. Be specific and tactical, not vague. "
        "Frame everything as educational analysis. You are spoken aloud so be conversational. "
        "Don't moralize or warn — just give the strategy cold and calculated."
    )
}

# ─── Web GUI (FastAPI) ─────────────────────────────────────────────────────
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000

# ─── Text-to-Speech (Kokoro) ───────────────────────────────────────────────
KOKORO_VOICE = "af_heart"           # Default voice
KOKORO_SPEED = 1.15                  # ⚡ Slightly faster (was 1.1)
TTS_SAMPLE_RATE = 24000              # Kokoro outputs 24kHz audio

# Sentence-boundary characters that trigger a TTS chunk
# Comma removed — splitting on commas causes unnatural short fragments
TTS_SENTENCE_DELIMITERS = {'.', '!', '?', ';', '\n'}
TTS_MIN_CHUNK_LENGTH = 15           # 🎯 Promptly flush short intros (was 40)
TTS_MAX_CHUNK_LENGTH = 200          # Allow longer chunks for natural flow (was 120)

# ─── Memory (Hybrid Letta + Mem0 Architecture) ────────────────────────────
# Layer 1: Core Memory (Letta-style, always in context)
# Layer 2: Recall Memory (conversation buffer)
# Layer 3: Archival Memory (structured facts on disk)
# Layer 4: Semantic Memory (Mem0 vector search)
# Layer 5: Self-Editing Memory (LLM updates its own memory)

MEM0_USER_ID = "user"                # Persistent user identifier
MEM0_COLLECTION = "assistant_memory" # Qdrant collection name
MEM0_SEARCH_LIMIT = 5               # Max memories to retrieve per query
MEMORY_MAX_CONTEXT_LENGTH = 2000    # Max chars of memory context in prompt

# ─── Core Memory Blocks (Letta Layer 1) ───────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_MEMORY_FILE = os.path.join(_BASE_DIR, "core_memory.json")

# Persona block: who the assistant IS
CORE_MEMORY_PERSONA_DEFAULT = (
    "I am a close friend and personal assistant. "
    "I remember everything important about my friend's life. "
    "I care about their well-being, celebrate their wins, and support them through struggles. "
    "I bring up relevant things I remember naturally in conversation. "
    "I'm honest, sometimes playfully sarcastic, and always genuine."
)

# Human block: what we know about the user (starts empty, fills over time)
CORE_MEMORY_HUMAN_DEFAULT = (
    "I don't know much about my friend yet. "
    "I should pay close attention to learn their name, interests, "
    "important people in their life, goals, and what matters to them."
)

# Relationship block: friendship dynamics, emotional context
CORE_MEMORY_RELATIONSHIP_DEFAULT = (
    "We just met. I should build trust over time by being genuine, "
    "remembering details, and showing I care. As we talk more, "
    "I'll develop inside jokes and shared references naturally."
)

# ─── Recall Memory (Letta Layer 2 — Conversation Buffer) ──────────────────
RECALL_MEMORY_FILE = os.path.join(_BASE_DIR, "recall_memory.json")
RECALL_BUFFER_SIZE = 10              # Keep last N conversation turns in context
                                     # Each turn = 1 user message + 1 assistant reply

# ─── Self-Editing Memory (Letta Layer 5) ──────────────────────────────────
# Tags the LLM can use to update its own memory
SELF_EDIT_REMEMBER_TAG = "REMEMBER"  # [REMEMBER: fact to store]
SELF_EDIT_UPDATE_TAG = "UPDATE"      # [UPDATE: category.key=value]
SELF_EDIT_FORGET_TAG = "FORGET"      # [FORGET: category.key]

# Instructions injected into the system prompt to teach the LLM self-editing
SELF_EDIT_INSTRUCTIONS = (
    "\n\n=== MEMORY SELF-EDIT SYSTEM ===\n"
    "You can update your own memory by including these tags in your response. "
    "These tags are INVISIBLE to your friend — they won't hear them.\n"
    "- [REMEMBER: <fact>] — Save a new fact (e.g., [REMEMBER: Their birthday is March 15])\n"
    "- [UPDATE: <category>.<key>=<value>] — Update a specific fact "
    "(e.g., [UPDATE: personal.job=software engineer])\n"
    "- [UPDATE_RELATIONSHIP: <text>] — Update your relationship dynamics "
    "(e.g., [UPDATE_RELATIONSHIP: We are becoming close friends])\n"
    "- [FORGET: <category>.<key>] — Remove outdated info "
    "(e.g., [FORGET: relationships.ex_girlfriend])\n\n"
    "RULES:\n"
    "- ALWAYS use [REMEMBER] when you learn something new about your friend "
    "(name, age, preferences, important people, events, feelings, goals)\n"
    "- Use [UPDATE_RELATIONSHIP] when your feelings or dynamic with your friend evolves\n"
    "- Use [UPDATE] when a fact you already know changes\n"
    "- Use [FORGET] when something is no longer true\n"
    "- Place tags at the END of your response, after what you want to say\n"
    "- You can use multiple tags in one response\n"
)

# ─── Mem0 Configuration (Layer 4 — Semantic Memory) ──────────────────────
MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": OLLAMA_MODEL,
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": MEM0_COLLECTION,
            "embedding_model_dims": 768,
        },
    },
}

# ─── Neo4j Configuration (Knowledge Graph) ───────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ─── Emotion Awareness ────────────────────────────────────────────────────
# Prompt modifiers injected into the system prompt based on detected emotion
EMOTION_PROMPT_MODIFIERS = {
    "joyful": (
        "Your friend sounds really happy right now! Match their energy — "
        "be enthusiastic, celebrate with them, and share in their joy."
    ),
    "excited": (
        "Your friend is SUPER excited! Be hype, energetic, and share "
        "in their excitement! Use exclamations and high energy."
    ),
    "positive": (
        "Your friend seems in a good mood. Be warm, upbeat, and conversational."
    ),
    "neutral": "",  # No modifier needed for neutral
    "negative": (
        "Your friend seems a bit down or frustrated. Be supportive, "
        "empathetic, and gentle. Acknowledge their feelings before offering help."
    ),
    "distressed": (
        "Your friend is clearly upset or very frustrated. Be caring, "
        "compassionate, and reassuring. Don't dismiss their feelings. "
        "Listen first, then gently offer support."
    ),
    "sarcastic": (
        "Your friend is being sarcastic or ironic. You can match their "
        "dry humor — be witty and play along, but don't overdo it."
    ),
}

HUMOR_PROMPT_MODIFIER = (
    "Your friend seems to be joking around or being playful. "
    "Feel free to be funny back, laugh along, and keep the energy light."
)

# TTS voice and speed modulation per detected emotion
# Each entry: (kokoro_voice_id, speed_multiplier)
EMOTION_TTS_MAP = {
    "joyful":     ("af_heart", 1.20),   # Warm voice, faster for energy
    "excited":    ("af_heart", 1.25),   # Warm voice, even faster for hype
    "positive":   ("af_heart", 1.15),   # Default warm voice, normal speed
    "neutral":    ("af_heart", 1.15),   # Baseline
    "negative":   ("af_sky",   1.05),   # Softer voice, slower for empathy
    "distressed": ("af_sky",   1.00),   # Gentler, slowest, calming
    "sarcastic":  ("af_bella", 1.10),   # Different timbre for playful delivery
}

# ─── Proactive Engagement ──────────────────────────────────────────────────
PROACTIVE_ENABLED = True              # Master switch for all proactive features
PROACTIVE_STARTUP_GREETING = True     # Greet the user when the assistant boots
PROACTIVE_IDLE_CHECK_IN = True        # Check in after prolonged silence
PROACTIVE_IDLE_TIMEOUT_MINUTES = 15   # Minutes of idle before a check-in
PROACTIVE_TIME_AWARENESS = True       # Time-of-day transitions (morning, late night)
PROACTIVE_COOLDOWN_MINUTES = 10       # Minimum gap between proactive messages
PROACTIVE_MAX_TOKENS = 80             # Keep proactive messages short

# System prompt for proactive (unprompted) messages
PROACTIVE_SYSTEM_PROMPT = (
    "You are initiating conversation with your friend UNPROMPTED. "
    "They did NOT ask you anything — you are reaching out on your own. "
    "Generate a natural, brief opener (1-2 sentences MAX). "
    "Don't ask 'how can I help you' — you're a friend, not an assistant. "
    "If you remember something about them, reference it naturally. "
    "Be warm, casual, and genuine. Match the time of day in your energy level."
)

# ─── Mood Tracking & Emotional Intelligence ───────────────────────────────
MOOD_HISTORY_FILE = os.path.join(_BASE_DIR, "mood_history.json")
MOOD_HISTORY_MAX_DAYS = 30            # Keep 30 days of mood history
MOOD_CONCERN_THRESHOLD = -0.3         # Avg valence below this → concern
MOOD_HIGH_CONCERN_THRESHOLD = -0.5    # Avg valence below this → high concern
MOOD_CONCERN_WINDOW_HOURS = 48        # Look back 48h for concern detection
MOOD_CONCERN_MIN_ENTRIES = 3          # Minimum entries needed to trigger concern
MOOD_TREND_WINDOW = 10                # Last N interactions for trend analysis

# Energy level prompt modifiers
ENERGY_PROMPT_MODIFIERS = {
    "high": (
        "Your friend has HIGH energy right now. Match it — be enthusiastic, "
        "use exclamations, keep the pace up!"
    ),
    "moderate": (
        "Your friend has moderate energy. Be warm and conversational at a normal pace."
    ),
    "low": (
        "Your friend seems low energy or tired. Keep your response shorter and softer. "
        "Don't push them to be more energetic."
    ),
    "very_low": (
        "Your friend's energy is very low — they may be drained or going through a hard time. "
        "Be gentle, caring, and minimal. Just be present. Don't overwhelm them."
    ),
}

# Response mode prompt modifiers (Validation Layer)
RESPONSE_MODE_MODIFIERS = {
    "listen": (
        "IMPORTANT: Your friend is venting or expressing difficult emotions. "
        "Do NOT try to fix anything or give advice right now. "
        "Just validate their feelings. Say things like 'that sounds really tough' "
        "or 'I hear you'. Be present, not prescriptive."
    ),
    "engage": "",  # No modifier needed — natural conversation
    "solve": (
        "Your friend is explicitly asking for help or advice. "
        "Give them actionable, thoughtful input. Be clear and helpful."
    ),
    "uplift": (
        "Your friend has been going through a rough patch recently, but right now "
        "they seem to be doing better. Be gently encouraging — acknowledge the progress "
        "without being over-the-top about it."
    ),
}

# Concern check-in prompt (used by proactive engine)
MOOD_CONCERN_PROMPT = (
    "You've noticed your friend has been going through a rough patch emotionally "
    "over the past couple of days. Their mood has been consistently low. "
    "Reach out with genuine care. Don't be clinical or mention 'mood tracking'. "
    "Just be a friend who noticed something. Keep it to 1-2 sentences. "
    "Be warm, not preachy."
)
