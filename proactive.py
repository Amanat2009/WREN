"""
proactive.py — Proactive Engagement Engine.

Makes the girlfriend talk on her own — startup greetings, post-conversation
follow-ups, and time-based observations. Uses memory context for personalized openers.

Runs as a daemon thread alongside the main wake-word loop.
"""

import logging
import random
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
        "morning":    f"It's {time_str} on {day_str}. Morning energy — she might be playfully awake.",
        "afternoon":  f"It's {time_str} on {day_str}. Afternoon — casual, teasing energy.",
        "evening":    f"It's {time_str} on {day_str}. Evening — warm, slightly more intimate energy.",
        "night":      f"It's {time_str} on {day_str}. Night — soft, affectionate, maybe a little clingy.",
        "late_night": f"It's {time_str} on {day_str}. Very late — she's still up and thinking about him.",
    }
    return hints.get(period, f"It's {time_str} on {day_str}.")


# ─── Varied Trigger Types ──────────────────────────────────────────────────
# Each trigger type generates a different flavor of proactive message.
# Randomly selected to keep things fresh and unpredictable.

_TRIGGER_TYPES = [
    "follow_up",      # Picks up on what was last talked about
    "miss_you",       # Affectionate "you got quiet on me" opener
    "random_thought", # Shares something she randomly "thought of"
    "teasing",        # Playfully provokes him to respond
    "internet_discovery", # Found something interesting on the web
    "find_activity",  # Propose a fun activity or game to do together
]

_TRIGGER_PROMPTS = {
    "follow_up": (
        "You remember what he was just talking about or doing. "
        "Follow up on it naturally — ask how it went, or make a comment about it. "
        "Keep it to 1-2 sentences. Be curious and warm."
    ),
    "miss_you": (
        "He went quiet on you and you noticed. "
        "Reach out in an affectionate, slightly pouty way — like a girlfriend who "
        "wants attention but is trying to play it cool. 1-2 sentences max."
    ),
    "random_thought": (
        "Spontaneously share something you 'just thought of' for no reason. "
        "It could be about him, about something he mentioned before, or just "
        "something random that reminded you of him. Keep it short and natural."
    ),
    "teasing": (
        "Tease him playfully to get him talking again. "
        "Could be a light jab, a flirty dare, or a provocative question. "
        "Be fun and a little mischievous. 1-2 sentences max."
    ),
    "internet_discovery": (
        "You got a bit bored and started browsing the internet. You found something interesting. "
        "Bring it up with him in a fun, entertaining way to spark a conversation. "
        "Keep it to 2-3 sentences. Here is what you found: \n\n[SEARCH_RESULTS]"
    ),
    "find_activity": (
        "You were thinking about fun things you two could do together. "
        "You looked up some ideas and found something cool. Pitch the idea to him playfully. "
        "Keep it to 2-3 sentences max. Here is what you found: \n\n[SEARCH_RESULTS]"
    ),
}


class ProactiveEngine:
    """
    Background engine that generates unprompted messages.

    Trigger types:
      1. Startup greeting — on boot, after all engines are ready
      2. Post-conversation follow-ups — fires shortly after silence begins
      3. Time-based — when time-of-day transitions (late night nudge, etc.)
      4. Concern check-in — when mood has been consistently low

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
        """Main proactive loop — checks triggers every 10 seconds."""
        # Step 1: Startup greeting (immediate, one-time)
        if config.PROACTIVE_STARTUP_GREETING:
            # Wait a moment for everything to settle after boot
            self._stop_event.wait(timeout=3.0)
            if not self._stop_event.is_set():
                self._do_startup_greeting()

        # Step 2: Continuous monitoring loop — polls every 10 seconds
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=10.0)

            if self._stop_event.is_set():
                break

            # Don't trigger if assistant is actively interacting
            if shared_state.current_status != "idle":
                continue

            # Respect cooldown between proactive messages
            if not self._cooldown_elapsed():
                continue

            # Check idle timeout — fire a varied post-conversation trigger
            if config.PROACTIVE_IDLE_CHECK_IN and self._is_idle_timeout():
                self._do_post_conversation_trigger()
                continue

            # Check mood concern trigger
            if self.mood:
                concern = self.mood._get_concern()
                if concern in ("concern", "high_concern"):
                    # Only trigger concern check-in every 2 hours max
                    if time.time() - self._last_concern_check >= 2 * 3600:
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

    def _pick_trigger(self) -> str:
        """Randomly select a trigger type to keep messages varied and natural."""
        return random.choice(_TRIGGER_TYPES)

    def _do_startup_greeting(self):
        """Generate and speak a personalized startup greeting."""
        logger.info("👋 Proactive: Generating startup greeting...")

        time_hint = _get_time_greeting()
        memory_ctx = self._get_memory_context("greeting startup hello")

        # Check for anniversary milestone
        anniversary_ctx = ""
        try:
            anniversary_ctx = self.memory.core.get_anniversary_context()
        except Exception:
            pass

        # Check last session mood for carryover tone
        last_mood_ctx = ""
        try:
            last_mood, _ = self.memory.core.get_last_session_mood()
            if last_mood in ("upset", "jealous"):
                last_mood_ctx = (
                    f"Last session you were feeling {last_mood}. "
                    f"You can still be a little cool at the start — "
                    f"don't fully erase that, but don't dwell on it either."
                )
        except Exception:
            pass

        prompt = (
            f"[CONTEXT] {time_hint}\n"
            f"{memory_ctx}\n"
            f"{anniversary_ctx}\n"
            f"{last_mood_ctx}\n\n"
            f"You just woke up and he's here. Greet him warmly and flirtatiously. "
            f"If you know something about him, reference it briefly. 1-2 sentences."
        )

        self._speak_proactive(prompt, label="startup_greeting")

    def _do_post_conversation_trigger(self):
        """Fire a varied post-conversation message using a randomly picked trigger type."""
        trigger = self._pick_trigger()
        idle_seconds = int(time.time() - shared_state.last_interaction_time)
        logger.info(f"💬 Proactive: Firing '{trigger}' after {idle_seconds}s of silence...")

        time_hint = _get_time_greeting()

        # Pull the last conversation turn for context if available
        recent_context = ""
        if self.memory.recall.turns:
            last_turn = self.memory.recall.turns[-1]
            last_user = last_turn.get("user", "")
            last_reply = last_turn.get("assistant", "")
            if last_user and not last_user.startswith("[PROACTIVE"):
                recent_context = (
                    f"[LAST CONVERSATION]\n"
                    f"He said: \"{last_user[:120]}\"\n"
                    f"You replied: \"{last_reply[:120]}\"\n"
                )

        # For follow_up, also inject a recent moment if available
        moments_ctx = ""
        if trigger == "follow_up":
            try:
                moments = self.memory.core.get_moments()
                if moments:
                    moments_ctx = f"[A MOMENT YOU REMEMBER] {moments[-1]}\n"
            except Exception:
                pass

        memory_ctx = self._get_memory_context("checking in thinking about you")

        trigger_prompt = _TRIGGER_PROMPTS[trigger]
        if trigger in ["internet_discovery", "find_activity"]:
            search_results = self._perform_random_search(trigger)
            if search_results:
                trigger_prompt = trigger_prompt.replace("[SEARCH_RESULTS]", search_results)
            else:
                # Fallback if search fails
                trigger_prompt = _TRIGGER_PROMPTS["random_thought"]
                logger.debug(f"⚠️ Proactive search empty, falling back to random thought")

        prompt = (
            f"[CONTEXT] {time_hint}\n"
            f"{recent_context}"
            f"{moments_ctx}"
            f"{memory_ctx}\n\n"
            f"{trigger_prompt}"
        )

        self._speak_proactive(prompt, label=trigger)

    def _perform_random_search(self, trigger: str) -> str:
        """Perform a random background search to fuel proactive conversation."""
        if not getattr(config, "TAVILY_API_KEY", None):
            return ""

        queries = {
            "internet_discovery": [
                "latest weird news",
                "interesting space discoveries today",
                "cool new tech gadgets",
                "fun pop culture news",
                "bizarre history facts"
            ],
            "find_activity": [
                "fun couple activities at home",
                "creative discussion topics for couples",
                "new co-op video games",
                "virtual experiences for couples",
                "fun two player board games"
            ]
        }
        
        query_list = queries.get(trigger)
        if not query_list:
            return ""

        query = random.choice(query_list)
        logger.info(f"🌐 Proactive background search: {query}")
        
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=config.TAVILY_API_KEY)
            res = client.search(
                query=query,
                max_results=2,
                search_depth="basic"
            )
            
            context_strs = []
            for item in res.get('results', []):
                context_strs.append(f"- {item.get('title', '')}: {item.get('content', '')}")
            
            return "\n".join(context_strs)
        except Exception as e:
            logger.debug(f"⚠️ Proactive search failed: {e}")
            return ""

    def _do_time_transition(self, old_period: str, new_period: str):
        """Generate a time-of-day observation."""
        logger.info(f"🕐 Proactive: Time transition {old_period} → {new_period}")

        time_hint = _get_time_greeting()
        memory_ctx = self._get_memory_context(f"time of day {new_period}")

        prompt = (
            f"[CONTEXT] {time_hint}\n"
            f"The time just shifted from {old_period} to {new_period}.\n"
            f"{memory_ctx}\n\n"
            f"Make a brief, natural girlfriend-style observation about the time. "
            f"Could be checking if he's eaten, teasing him about being up late, etc."
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

            # Build system prompt — safely fall back to first available personality
            personality_key = shared_state.current_personality
            fallback_key = list(config.LLM_PERSONALITIES.keys())[0]
            personality = config.LLM_PERSONALITIES.get(
                personality_key, config.LLM_PERSONALITIES[fallback_key]
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
            print(f"\n   💬 [Proactive/{label}] {clean_text}")

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
