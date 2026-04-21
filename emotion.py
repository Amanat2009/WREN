"""
emotion.py — Transformer-based emotion detection for the voice assistant.

Uses j-hartmann/emotion-english-distilroberta-base for accurate
7-class emotion classification. Runs on CPU with <300ms latency
per inference on an i5-10400F.

Model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
These are mapped to our system categories for TTS and prompt modulation.

Humor and sarcasm cues are detected via fast regex (essentially 0ms)
since transformers aren't reliable for those.
"""

import logging
import re
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ─── Model configuration ───────────────────────────────────────────────────
_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
_pipeline = None
_pipeline_lock = threading.Lock()
_ready = threading.Event()


# ─── Humor detection patterns (regex-based, ~0ms) ──────────────────────────
_HUMOR_PATTERNS = re.compile(
    r'\b(lol|lmao|rofl|haha|hehe|just kidding|jk|joke|funny|hilarious)\b',
    re.IGNORECASE,
)

# Sarcasm lexical cues (transformers miss these reliably)
_SARCASM_STARTERS = re.compile(
    r'^(oh great|oh wow|oh sure|oh yeah|oh fantastic|wow thanks'
    r'|yeah right|sure thing|how wonderful|how lovely)',
    re.IGNORECASE,
)

# Intensity cues for excited vs joyful
_EXCLAMATION_RE = re.compile(r'!{2,}')
_ALL_CAPS_WORD_RE = re.compile(r'\b[A-Z]{3,}\b')


# ─── Map transformer labels → our system categories ──────────────────────
# Model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
_LABEL_MAP = {
    "joy":      "joyful",
    "neutral":  "neutral",
    "sadness":  "negative",
    "anger":    "distressed",
    "fear":     "negative",
    "surprise": "positive",
    "disgust":  "negative",
}

# Confidence thresholds for stronger category variants
_DISTRESSED_THRESHOLD = 0.6   # sadness above this → distressed
_EXCITED_THRESHOLD = 0.5      # joy above this + intensity cues → excited
_JOYFUL_FLOOR = 0.5           # joy below this → "positive" instead of "joyful"


@dataclass
class EmotionResult:
    """Result of emotion analysis on user text."""
    category: str          # joyful, excited, positive, neutral, negative, distressed, sarcastic
    compound: float        # Top emotion confidence (0.0 to 1.0)
    intensity: float       # Normalized intensity (0.0 to 1.0)
    is_humor: bool         # Whether humor/joke cues are detected
    details: dict          # All label→score from the transformer


def warmup():
    """
    Pre-load the transformer emotion model.

    Called during the parallel warmup phase in Assistant.__init__
    so the model is ready before the first user interaction.
    First run downloads ~330MB; subsequent runs load from cache (~1-2s).
    """
    global _pipeline
    import os
    logger.info(f"🔥 Loading emotion model ({_MODEL_NAME})...")
    try:
        from transformers import pipeline as tf_pipeline

        # ⚡ Suppress huggingface_hub network calls (metadata, discussions,
        # commits, PR threads) when the model is already cached locally.
        # On first run (no cache), remove this or it will fail to download.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        with _pipeline_lock:
            _pipeline = tf_pipeline(
                "text-classification",
                model=_MODEL_NAME,
                top_k=None,       # Return all 7 label scores
                device=-1,        # Force CPU (no ROCm on RX 6500 XT)
                truncation=True,  # Auto–truncate to 512 tokens
            )

        # Run a single dummy inference to fully JIT the model
        _pipeline("warm up")

        _ready.set()
        logger.info("✅ Emotion model loaded and warmed up.")
    except Exception as e:
        _ready.set()  # Unblock callers even on failure
        logger.warning(f"⚠️  Emotion model failed to load: {e}")


def analyze(text: str) -> EmotionResult:
    """
    Classify the emotional content of user speech.

    Pipeline:
      1. Fast regex for humor + sarcasm cues (~0ms)
      2. Transformer inference for core emotion (~100-200ms CPU)
      3. Post-processing: map labels, apply intensity refinements

    Returns:
        EmotionResult with category, confidence, intensity, humor flag.
    """
    default = EmotionResult(
        category="neutral", compound=0.0, intensity=0.0,
        is_humor=False, details={},
    )

    if not text or not text.strip():
        return default

    # ── Step 1: Fast regex checks (essentially free) ──
    is_humor = bool(_HUMOR_PATTERNS.search(text))
    has_exclamations = bool(_EXCLAMATION_RE.search(text))
    caps_count = len(_ALL_CAPS_WORD_RE.findall(text))
    has_intensity = has_exclamations or caps_count >= 2
    is_sarcastic_cue = bool(_SARCASM_STARTERS.search(text.strip()))

    # ── Step 2: Wait for model readiness ──
    if not _ready.is_set():
        _ready.wait(timeout=10)

    if _pipeline is None:
        logger.debug("Emotion model not available, defaulting to neutral.")
        default.is_humor = is_humor
        return default

    # ── Step 3: Transformer inference ──
    try:
        raw = _pipeline(text[:512])

        # Parse scores → {label: score}
        scores = {}
        items = raw[0] if raw and isinstance(raw[0], list) else raw
        for item in items:
            scores[item["label"]] = item["score"]

        if not scores:
            default.is_humor = is_humor
            return default

        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]

        # ── Step 4: Map to our category system ──
        category = _LABEL_MAP.get(top_label, "neutral")

        # ── Corrections for known model quirks ──

        # ALL-CAPS + exclamations can confuse the model into "anger".
        # If anger is top but not dominant (< 0.5), and the text has
        # strong positive intensity cues (caps AND exclamations),
        # override to excited — this is celebration, not rage.
        if top_label == "anger" and has_intensity and top_score < 0.5:
            if has_exclamations and caps_count >= 2:
                # Very strong intensity (caps + exclamations) → excited
                category = "excited"
                top_score = max(scores.get("joy", 0) + scores.get("surprise", 0), top_score)
            elif scores.get("joy", 0) + scores.get("surprise", 0) > top_score:
                # Combined positive signals beat anger → excited
                category = "excited"
                top_score = scores.get("joy", 0) + scores.get("surprise", 0)

        # Sarcasm override (lexical cue — transformer can't detect this)
        elif is_sarcastic_cue:
            category = "sarcastic"

        # High-confidence sadness → distressed
        elif top_label == "sadness" and top_score >= _DISTRESSED_THRESHOLD:
            category = "distressed"

        # Anger with high confidence is genuinely distressed
        elif top_label == "anger" and top_score >= 0.5:
            category = "distressed"

        # Joy with intensity cues → excited
        elif top_label == "joy" and has_intensity and top_score >= _EXCITED_THRESHOLD:
            category = "excited"

        # Moderate joy (low confidence) → positive, not joyful
        elif top_label == "joy" and top_score < _JOYFUL_FLOOR:
            category = "positive"

        # Surprise with intensity → excited
        elif top_label == "surprise" and has_intensity:
            category = "excited"

        # ── Compute intensity ──
        intensity = min(top_score, 1.0)
        if has_intensity:
            intensity = min(intensity * 1.2, 1.0)

        result = EmotionResult(
            category=category,
            compound=top_score,
            intensity=intensity,
            is_humor=is_humor,
            details=scores,
        )

        logger.debug(
            f"🎭 Emotion: {category} (model={top_label}:{top_score:.3f}, "
            f"humor={is_humor})"
        )
        return result

    except Exception as e:
        logger.warning(f"⚠️  Emotion analysis failed: {e}")
        default.is_humor = is_humor
        return default


# ─── Girlfriend Response Emotion Analyzer ─────────────────────────────────
# Analyzes the LLM's OWN generated reply to determine her emotional state.
# This drives TTS prosody (voice, speed) from HER feelings — not the user's.
# Uses fast regex (zero model inference) — runs in <1ms.

_INTIMATE_RE = re.compile(
    r'\b(i love you|love you|miss you|thinking about you|cuddle|hold you|'
    r'kiss|babe|baby|sweetheart|darling|forever|belong to me|mine|'
    r'close to you|next to you)\b',
    re.IGNORECASE,
)
_JEALOUS_RE = re.compile(
    r'\b(who is she|who is he|another girl|other girl|other woman|'
    r'talking to her|texting her|been with|hanging out with|'
    r'why were you|don\'t you dare|excuse me|i saw that|'
    r'you\'re mine|back off|stay away|i don\'t like|not okay)\b',
    re.IGNORECASE,
)
_PLAYFUL_RE = re.compile(
    r'\b(haha|lol|lmao|teasing|just kidding|jk|wink|😏|😜|🙈|'
    r'bet you|i dare you|catch me|try me|come on|oh please|'
    r'you wish|whatever|rolling my eyes)\b',
    re.IGNORECASE,
)
_EXCITED_RE = re.compile(
    r'(!{2,}|omg|oh my god|no way|seriously\?|wait what|'
    r'i can\'t believe|that\'s amazing|oh wow|yesss|yasss)',
    re.IGNORECASE,
)
_UPSET_RE = re.compile(
    r'\b(you ignored|you didn\'t|you forgot|you never|you always|'
    r'i\'m annoyed|i\'m mad|i\'m angry|that\'s not fair|'
    r'you don\'t care|fine|whatever then|i\'m done|leave me alone)\b',
    re.IGNORECASE,
)
_SOFT_RE = re.compile(
    r'\b(it\'s okay|i\'m here|don\'t worry|you\'re okay|'
    r'i got you|take your time|breathe|i understand|'
    r'that must be hard|i care about you|i\'m sorry you feel)\b',
    re.IGNORECASE,
)


def analyze_response(text: str) -> str:
    """
    Classify the girlfriend's OWN generated reply into an emotional state.
    Uses fast regex patterns — zero model inference (<1ms).

    Returns one of:
        "intimate", "jealous", "playful", "excited", "upset", "soft", "neutral"
    """
    if not text or not text.strip():
        return "neutral"

    # Priority order matters — jealous/upset trump playful, intimate trumps neutral
    if _JEALOUS_RE.search(text):
        logger.debug("🎭 GF Emotion: jealous")
        return "jealous"
    if _UPSET_RE.search(text):
        logger.debug("🎭 GF Emotion: upset")
        return "upset"
    if _INTIMATE_RE.search(text):
        logger.debug("🎭 GF Emotion: intimate")
        return "intimate"
    if _SOFT_RE.search(text):
        logger.debug("🎭 GF Emotion: soft")
        return "soft"
    if _EXCITED_RE.search(text):
        logger.debug("🎭 GF Emotion: excited")
        return "excited"
    if _PLAYFUL_RE.search(text):
        logger.debug("🎭 GF Emotion: playful")
        return "playful"

    logger.debug("🎭 GF Emotion: neutral")
    return "neutral"
