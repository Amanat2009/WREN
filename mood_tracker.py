"""
mood_tracker.py — Persistent mood tracking and emotional intelligence.

Tracks the user's emotional state over time, computes trends,
detects rough patches, determines energy levels, and decides
whether to listen, engage, solve, or uplift.

Data persisted to mood_history.json. All analysis is in-memory
arithmetic — <1ms per call.
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# ─── Valence mapping from emotion categories ──────────────────────────────
# Positive emotions → positive valence, negative → negative
_CATEGORY_VALENCE = {
    "excited":    0.9,
    "joyful":     0.7,
    "positive":   0.4,
    "neutral":    0.0,
    "sarcastic":  0.1,   # Mild positive — sarcasm is usually playful
    "negative":  -0.4,
    "distressed": -0.8,
}

# ─── Question detection for solve mode ────────────────────────────────────
_QUESTION_RE = re.compile(
    r'\?|'
    r'^(how|what|why|should|can you|could you|help me|tell me|explain|do you think)',
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class MoodEntry:
    """A single mood data point."""
    ts: float              # Unix timestamp
    date: str              # YYYY-MM-DD for calendar-day grouping
    category: str          # Emotion category
    compound: float        # Transformer confidence
    valence: float         # Signed mood score (-1.0 to 1.0)
    is_humor: bool
    preview: str           # First 50 chars of user text


@dataclass
class MoodAnalysis:
    """Complete mood intelligence for a single interaction."""
    trend: str             # "improving", "stable", "declining"
    energy: str            # "high", "moderate", "low", "very_low"
    concern: str           # "none", "concern", "high_concern"
    response_mode: str     # "listen", "engage", "solve", "uplift"
    avg_valence: float     # Rolling average valence
    entry_count: int       # Total entries in history
    summary: str           # Human-readable mood summary for the LLM


class MoodTracker:
    """
    Persistent mood tracking with trend analysis and concern detection.

    Thread-safe. Data stored as JSON on disk, loaded into memory.
    All analysis functions are pure arithmetic — <1ms each.
    """

    def __init__(self, filepath: str = config.MOOD_HISTORY_FILE):
        self.filepath = filepath
        self._lock = threading.Lock()
        self.entries: List[MoodEntry] = self._load()
        self._prune_old_entries()
        logger.info(f"📊 Mood tracker loaded ({len(self.entries)} entries)")

    def _load(self) -> List[MoodEntry]:
        """Load mood history from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [MoodEntry(**e) for e in data]
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.warning(f"⚠️  Failed to load mood history: {e}")
        return []

    def _save(self):
        """Save mood history to disk."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(e) for e in self.entries],
                    f, indent=2, ensure_ascii=False,
                )
        except IOError as e:
            logger.error(f"❌ Failed to save mood history: {e}")

    def _prune_old_entries(self):
        """Remove entries older than MOOD_HISTORY_MAX_DAYS."""
        cutoff = time.time() - (config.MOOD_HISTORY_MAX_DAYS * 86400)
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.ts >= cutoff]
        pruned = before - len(self.entries)
        if pruned > 0:
            logger.info(f"🗑️  Pruned {pruned} old mood entries.")
            self._save()

    # ─── Logging ──────────────────────────────────────────────────────────

    def log(self, emotion_result, user_text: str = ""):
        """
        Log a mood entry from an EmotionResult.

        Args:
            emotion_result: An EmotionResult from emotion.analyze()
            user_text: The user's transcribed speech (for preview)
        """
        # Compute valence from the transformer's raw label scores
        details = emotion_result.details or {}
        if details:
            pos_score = details.get("joy", 0) + details.get("surprise", 0)
            neg_score = (
                details.get("anger", 0) + details.get("sadness", 0)
                + details.get("fear", 0) + details.get("disgust", 0)
            )
            valence = pos_score - neg_score
        else:
            # Fallback to category-based valence
            valence = _CATEGORY_VALENCE.get(emotion_result.category, 0.0)

        now = datetime.now()
        entry = MoodEntry(
            ts=time.time(),
            date=now.strftime("%Y-%m-%d"),
            category=emotion_result.category,
            compound=emotion_result.compound,
            valence=round(valence, 3),
            is_humor=emotion_result.is_humor,
            preview=user_text[:50] if user_text else "",
        )

        with self._lock:
            self.entries.append(entry)
            self._save()

        logger.debug(
            f"📊 Mood logged: {entry.category} (valence={entry.valence:+.3f})"
        )

    # ─── Analysis functions ───────────────────────────────────────────────

    def analyze(self, current_emotion, user_text: str = "") -> MoodAnalysis:
        """
        Run full mood analysis for the current interaction.

        Combines trend, energy, concern, and response mode into
        a single MoodAnalysis object. <1ms total.

        Args:
            current_emotion: EmotionResult from emotion.analyze()
            user_text: The user's transcribed speech
        """
        trend, avg_valence = self._get_trend()
        energy = self._get_energy(current_emotion, avg_valence)
        concern = self._get_concern()
        response_mode = self._get_response_mode(
            current_emotion, concern, user_text
        )
        summary = self._build_summary(
            current_emotion, trend, energy, concern, response_mode, avg_valence
        )

        return MoodAnalysis(
            trend=trend,
            energy=energy,
            concern=concern,
            response_mode=response_mode,
            avg_valence=round(avg_valence, 3),
            entry_count=len(self.entries),
            summary=summary,
        )

    def _get_trend(self) -> Tuple[str, float]:
        """
        Compute rolling mood trend over the last N interactions.

        Returns:
            (trend_label, avg_valence)
        """
        window = config.MOOD_TREND_WINDOW
        with self._lock:
            recent = self.entries[-window:] if self.entries else []

        if len(recent) < 2:
            return "stable", 0.0

        avg = sum(e.valence for e in recent) / len(recent)

        # Compare first half vs second half for direction
        mid = len(recent) // 2
        first_half_avg = sum(e.valence for e in recent[:mid]) / max(mid, 1)
        second_half_avg = sum(e.valence for e in recent[mid:]) / max(len(recent) - mid, 1)

        diff = second_half_avg - first_half_avg

        if diff > 0.15:
            trend = "improving"
        elif diff < -0.15:
            trend = "declining"
        else:
            trend = "stable"

        return trend, avg

    def _get_energy(self, current_emotion, avg_valence: float) -> str:
        """
        Map current emotion + trend into an energy level.

        Uses the immediate emotion (reactive) blended with the
        rolling average (sustained state).
        """
        cat = current_emotion.category
        intensity = current_emotion.intensity

        # Immediate energy from current emotion
        if cat in ("excited", "joyful") and intensity > 0.5:
            immediate = "high"
        elif cat in ("positive", "sarcastic", "neutral"):
            immediate = "moderate"
        elif cat == "negative":
            immediate = "low"
        elif cat == "distressed":
            immediate = "very_low"
        else:
            immediate = "moderate"

        # Sustained energy from trend
        if avg_valence > 0.3:
            sustained = "high"
        elif avg_valence > 0.0:
            sustained = "moderate"
        elif avg_valence > -0.3:
            sustained = "low"
        else:
            sustained = "very_low"

        # Blend: immediate emotion takes priority, but sustained
        # can pull it down (not up — we don't fake energy)
        energy_order = ["very_low", "low", "moderate", "high"]
        imm_idx = energy_order.index(immediate)
        sus_idx = energy_order.index(sustained)

        # Use whichever is lower (more conservative)
        result_idx = min(imm_idx, sus_idx + 1)  # sustained can pull down by 1 level max
        result_idx = max(0, min(result_idx, len(energy_order) - 1))

        return energy_order[result_idx]

    def _get_concern(self) -> str:
        """
        Check if the user has been in a sustained negative state.

        Looks at the last MOOD_CONCERN_WINDOW_HOURS of entries.
        """
        cutoff = time.time() - (config.MOOD_CONCERN_WINDOW_HOURS * 3600)
        with self._lock:
            recent = [e for e in self.entries if e.ts >= cutoff]

        if len(recent) < config.MOOD_CONCERN_MIN_ENTRIES:
            return "none"

        avg_valence = sum(e.valence for e in recent) / len(recent)
        neg_count = sum(
            1 for e in recent
            if e.category in ("negative", "distressed")
        )
        neg_ratio = neg_count / len(recent)

        if avg_valence < config.MOOD_HIGH_CONCERN_THRESHOLD and neg_ratio > 0.8:
            return "high_concern"
        elif avg_valence < config.MOOD_CONCERN_THRESHOLD and neg_ratio > 0.6:
            return "concern"
        else:
            return "none"

    def _get_response_mode(
        self, current_emotion, concern: str, user_text: str
    ) -> str:
        """
        Determine the right emotional posture for the response.

        listen: user is venting, don't fix
        engage: normal conversation
        solve:  user is asking for help
        uplift: coming out of a rough patch
        """
        cat = current_emotion.category
        is_question = bool(_QUESTION_RE.search(user_text)) if user_text else False

        # If user explicitly asks a question → solve mode
        if is_question and cat not in ("distressed",):
            return "solve"

        # If user is distressed/negative and NOT asking a question → listen
        if cat in ("distressed", "negative") and not is_question:
            return "listen"

        # If concern flag is set but current mood is neutral/positive → uplift
        if concern in ("concern", "high_concern") and cat in (
            "neutral", "positive", "joyful"
        ):
            return "uplift"

        # Default: engage
        return "engage"

    def _build_summary(
        self, current_emotion, trend, energy, concern, response_mode, avg_valence
    ) -> str:
        """Build a human-readable mood summary for injection into the LLM prompt."""
        parts = []

        # Trend context
        if trend == "improving":
            parts.append(
                "Your friend's mood has been improving recently."
            )
        elif trend == "declining":
            parts.append(
                "Your friend's mood has been declining over recent conversations."
            )

        # Concern context
        if concern == "high_concern":
            parts.append(
                "IMPORTANT: Your friend has been consistently low/upset "
                "for the past couple of days. Be extra gentle and caring."
            )
        elif concern == "concern":
            parts.append(
                "Your friend has been a bit down over recent interactions. "
                "Keep this in mind."
            )

        # Humor note
        if current_emotion.is_humor:
            parts.append(
                "They seem to be in a playful/joking mood right now."
            )

        if not parts:
            return ""

        return "\n".join(parts)

    # ─── Stats for debugging / proactive engine ──────────────────────────

    def get_recent_stats(self, hours: int = 24) -> dict:
        """Get mood statistics for the last N hours."""
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            recent = [e for e in self.entries if e.ts >= cutoff]

        if not recent:
            return {"count": 0, "avg_valence": 0.0, "categories": {}}

        avg = sum(e.valence for e in recent) / len(recent)
        cats = {}
        for e in recent:
            cats[e.category] = cats.get(e.category, 0) + 1

        return {
            "count": len(recent),
            "avg_valence": round(avg, 3),
            "categories": cats,
        }
