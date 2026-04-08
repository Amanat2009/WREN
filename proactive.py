"""
proactive.py — Proactive Engagement Engine.

Makes the assistant talk on its own — startup greetings, idle check-ins,
and time-based observations. Uses memory context for personalized openers.

Runs as a daemon thread alongside the main wake-word loop.
"""

import logging
import time
import threading
from datetime import datetime
from typing import Optional

import config
import shared_state
import llm as llm_module

logger = logging.getLogger(__name__)


def _get_time_period() -> str:
    """Get a human-friendly time period label."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    elif 21 <= hour < 24:
        return "night"
    else:
        return "late_night"


def _get_time_greeting() -> str:
    """Get a time-appropriate greeting hint for the LLM prompt."""
    period = _get_time_period()
    now = datetime.now()
    time_str = now.strftime('%I:%M %p')
    day_str = now.strftime('%A, %B %d')

    hints = {
        "morning": f"It's {time_str} on {day_str}. Good morning energy.",
        "afternoon": f"It's {time_str} on {day_str}. Casual afternoon vibe.",
        "evening": f"It's {time_str} on {day_str}. Relaxed evening energy.",
        "night": f"It's {time_str} on {day_str}. Winding down, maybe check if they should rest.",
        "late_night": (
            f"It's {time_str} on {day_str}. It's very late! "
            "Gently mention they might want to get some sleep."
        ),
    }
    return hints.get(period, f"It's {time_str} on {day_str}.")


class ProactiveEngine:
    """
    Background engine that generates unprompted messages.

    Trigger types:
      1. Startup greeting — on boot, after all engines are ready
      2. Idle check-in — after N minutes of user silence
      3. Time-based — when time-of-day transitions (late night nudge, etc.)

    Thread-safe: acquires shared_state.interaction_lock before speaking,
    and checks shared_state.current_status to avoid collisions.
    """

    def __init__(self, tts_engine, memory_manager, mood_tracker=None):
        self.tts = tts_engine
        self.memory = memory_manager
        self.mood = mood_tracker
        self._stop_event = threading.Event()
        self._last_proactive_time = 0.0  # Timestamp of last proactive message
        self._last_time_period = _get_time_period()  # Track period transitions
        self._last_concern_check = 0.0  # Track concern check-ins separately
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the proactive engine daemon thread."""
        if not config.PROACTIVE_ENABLED:
            logger.info("⏸️  Proactive engagement disabled in config.")
            return

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("✅ Proactive engagement engine started.")

    def stop(self):
        """Signal the engine to stop."""
        self._stop_event.set()

    def _run_loop(self):
        """Main proactive loop — checks triggers every 30 seconds."""
        # Step 1: Startup greeting (immediate, one-time)
        if config.PROACTIVE_STARTUP_GREETING:
            # Wait a moment for everything to settle after boot
            self._stop_event.wait(timeout=3.0)
            if not self._stop_event.is_set():
                self._do_startup_greeting()

        # Step 2: Continuous monitoring loop
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=30.0)  # Check every 30s

            if self._stop_event.is_set():
                break

            # Don't trigger if assistant is actively interacting
            if shared_state.current_status != "idle":
                continue

            # Respect cooldown
            if not self._cooldown_elapsed():
                continue

            # Check idle timeout
            if config.PROACTIVE_IDLE_CHECK_IN and self._is_idle_timeout():
                self._do_idle_check_in()
                continue

            # Check mood concern trigger
            if self.mood:
                concern = self.mood._get_concern()
                if concern in ("concern", "high_concern"):
                    # Only trigger concern check-in every 4 hours max
                    if time.time() - self._last_concern_check >= 4 * 3600:
                        self._do_concern_check_in(concern)
                        continue

            # Check time-of-day transition
            if config.PROACTIVE_TIME_AWARENESS:
                current_period = _get_time_period()
                if current_period != self._last_time_period:
                    old_period = self._last_time_period
                    self._last_time_period = current_period
                    self._do_time_transition(old_period, current_period)

    def _cooldown_elapsed(self) -> bool:
        """Check if enough time has passed since the last proactive message."""
        elapsed = time.time() - self._last_proactive_time
        return elapsed >= config.PROACTIVE_COOLDOWN_MINUTES * 60

    def _is_idle_timeout(self) -> bool:
        """Check if the user has been idle for longer than the timeout."""
        elapsed = time.time() - shared_state.last_interaction_time
        return elapsed >= config.PROACTIVE_IDLE_TIMEOUT_MINUTES * 60

    def _do_startup_greeting(self):
        """Generate and speak a personalized startup greeting."""
        logger.info("👋 Proactive: Generating startup greeting...")

        time_hint = _get_time_greeting()

        # Pull memory context for personalization
        memory_ctx = self._get_memory_context("greeting startup hello")

        prompt = (
            f"[CONTEXT] {time_hint}\n"
            f"{memory_ctx}\n\n"
            f"Generate a warm startup greeting for your friend. "
            f"If you know something about them, reference it briefly."
        )

        self._speak_proactive(prompt, label="startup_greeting")

    def _do_idle_check_in(self):
        """Generate and speak a check-in after prolonged silence."""
        idle_minutes = (time.time() - shared_state.last_interaction_time) / 60
        logger.info(f"💬 Proactive: Idle check-in after {idle_minutes:.0f} min...")

        time_hint = _get_time_greeting()
        memory_ctx = self._get_memory_context("checking in how are you")

        prompt = (
            f"[CONTEXT] {time_hint}\n"
            f"Your friend has been quiet for about {int(idle_minutes)} minutes.\n"
            f"{memory_ctx}\n\n"
            f"Check in casually. Don't be clingy. Just a friendly poke."
        )

        self._speak_proactive(prompt, label="idle_check_in")

    def _do_time_transition(self, old_period: str, new_period: str):
        """Generate a time-of-day observation."""
        logger.info(f"🕐 Proactive: Time transition {old_period} → {new_period}")

        time_hint = _get_time_greeting()
        memory_ctx = self._get_memory_context(f"time of day {new_period}")

        prompt = (
            f"[CONTEXT] {time_hint}\n"
            f"The time just shifted from {old_period} to {new_period}.\n"
            f"{memory_ctx}\n\n"
            f"Make a brief, natural observation about the time."
        )

        self._speak_proactive(prompt, label="time_transition")

    def _do_concern_check_in(self, concern_level: str):
        """Generate and speak a caring check-in when mood has been low."""
        logger.info(f"💜 Proactive: Concern check-in (level={concern_level})")

        time_hint = _get_time_greeting()
        memory_ctx = self._get_memory_context("how are you feeling concerned")

        stats = self.mood.get_recent_stats(hours=48) if self.mood else {}
        avg_valence = stats.get("avg_valence", 0.0)

        prompt = (
            f"[CONTEXT] {time_hint}\n"
            f"Concern level: {concern_level}. "
            f"Recent avg mood valence: {avg_valence:+.2f} (negative = bad).\n"
            f"{memory_ctx}\n\n"
            f"{config.MOOD_CONCERN_PROMPT}"
        )

        self._speak_proactive(prompt, label="concern_check_in")
        self._last_concern_check = time.time()

    def _get_memory_context(self, query: str) -> str:
        """Pull memory context for the proactive prompt."""
        try:
            ctx = self.memory.get_context_for_prompt(query)
            if ctx:
                return f"[MEMORY]\n{ctx}"
        except Exception as e:
            logger.debug(f"⚠️  Proactive memory fetch failed: {e}")
        return "[MEMORY] No specific memories to reference."

    def _speak_proactive(self, prompt: str, label: str = "proactive"):
        """Generate and speak a proactive message, with collision safety."""
        # Check status again right before speaking
        if shared_state.current_status != "idle":
            logger.info(f"⏩ Proactive [{label}] skipped — assistant is busy.")
            return

        # Try to acquire the interaction lock
        acquired = shared_state.interaction_lock.acquire(timeout=1.0)
        if not acquired:
            logger.info(f"⏩ Proactive [{label}] skipped — lock busy.")
            return

        try:
            # Double-check status after acquiring lock
            if shared_state.current_status != "idle":
                logger.info(f"⏩ Proactive [{label}] skipped — status changed.")
                return

            shared_state.current_status = "speaking"
            shared_state.current_response_text = ""

            # Build system prompt
            personality_key = shared_state.current_personality
            personality = config.LLM_PERSONALITIES.get(
                personality_key, config.LLM_PERSONALITIES["unfiltered"]
            )
            system = personality + "\n\n" + config.PROACTIVE_SYSTEM_PROMPT

            # Generate response via LLM (non-streaming for simplicity)
            full_text = ""
            for chunk in llm_module.stream_response(
                user_message=prompt,
                memory_context="",
                update_ui=False,
            ):
                full_text += chunk

            # Clean up self-edit tags if any leaked through
            from memory_manager import SelfEditEngine
            clean_text = SelfEditEngine.strip_tags(full_text).strip()

            if not clean_text:
                logger.info(f"⏩ Proactive [{label}] — LLM returned empty.")
                return

            # Update UI state
            shared_state.current_response_text = clean_text
            logger.info(f"💬 Proactive [{label}]: \"{clean_text[:80]}\"")
            print(f"\n   💬 [Proactive] {clean_text}")

            # Speak via TTS
            def _gen():
                yield clean_text

            self.tts.speak_stream(_gen())

            # Record this proactive message in recall memory
            self.memory.recall.add_turn(
                f"[PROACTIVE:{label}]", clean_text
            )

            # Update timestamps
            self._last_proactive_time = time.time()
            shared_state.last_interaction_time = time.time()

        except Exception as e:
            logger.error(f"❌ Proactive [{label}] error: {e}", exc_info=True)
        finally:
            shared_state.current_status = "idle"
            shared_state.interaction_lock.release()
