"""
assistant.py — Main orchestrator pipeline.
🚀 OPTIMIZED: parallel warmup, overlapped memory+STT, timing instrumentation.
Coordinates: wake word → BEEP → record → STT → LLM (streaming) → TTS (concurrent).
"""

import logging
import time
import threading
import winsound

from wake_word import WakeWordDetector
from stt import SpeechToText
from tts import TTSEngine
from memory_manager import MemoryManager
import audio_utils
import llm as llm_module
import config
import shared_state
import web_server

logger = logging.getLogger(__name__)


class Assistant:
    """
    Voice assistant orchestrator.

    Lifecycle:
        1. Initialize all engines (with parallel warmup)
        2. Enter idle loop (wake word detection)
        3. On wake → BEEP → record → transcribe → stream LLM → stream TTS
        4. Extract memory in background
        5. Return to idle
    """

    def __init__(self):
        self._running = False
        self._init_engines()

    def _init_engines(self):
        """Initialize all sub-engines with parallel warmup."""
        logger.info("=" * 60)
        logger.info("🚀 Initializing Voice Assistant...")
        logger.info("=" * 60)

        t0 = time.perf_counter()

        # Wake word detector
        self.wake_detector = WakeWordDetector()

        # Speech-to-text
        self.stt = SpeechToText()

        # Text-to-speech
        self.tts = TTSEngine()

        # Memory (Mem0 + Core memory — Mem0 inits in background)
        self.memory = MemoryManager()

        # ⚡ Warmup all engines in parallel to reduce startup time
        warmup_threads = [
            threading.Thread(target=self.stt.warmup, daemon=True),
            threading.Thread(target=self.tts.warmup, daemon=True),
            threading.Thread(target=llm_module.warmup, daemon=True),
        ]
        for t in warmup_threads:
            t.start()
        for t in warmup_threads:
            t.join(timeout=30)

        # 🌐 Start the Web UI Server
        web_server.start_server_in_background()

        # 🎧 Calibrate the microphone to the room's ambient noise floor
        audio_utils.calibrate_ambient_noise(duration_sec=1.0)

        elapsed = time.perf_counter() - t0
        logger.info(f"✅ All engines loaded + warmed up in {elapsed:.1f}s")
        logger.info("=" * 60)

    def _play_beep(self):
        """Play a short beep to acknowledge wake word detection."""
        try:
            winsound.Beep(config.BEEP_FREQUENCY, config.BEEP_DURATION)
        except Exception as e:
            logger.debug(f"⚠️  Beep failed (non-fatal): {e}")

    def run(self):
        """Main loop: idle → wake → beep → process → repeat."""
        self._running = True
        logger.info("🎧 Voice assistant is READY. Say the wake word to begin!")
        print("\n" + "=" * 50)
        print("  🎙️  LOCAL VOICE ASSISTANT")
        print("  Say 'Porcupine' to activate")
        print("  Press Ctrl+C to quit")
        print("=" * 50 + "\n")

        while self._running:
            try:
                # ── Reset state for new interaction ──
                shared_state.current_user_text = ""
                shared_state.current_response_text = ""
                shared_state.current_status = "idle"
                detected = self.wake_detector.wait_for_wake_word()
                if not detected or not self._running:
                    continue

                # ── Step 2: BEEP + Acknowledge wake ──
                shared_state.current_status = "listening"
                t_start = time.perf_counter()
                self._play_beep()
                print("\n🔔 Wake word detected! Listening...")
                logger.info("─" * 40)

                # ── Step 3: Record user speech ──
                shared_state.current_status = "recording"
                audio = audio_utils.record_speech()

                # Skip if recording is too short (noise/accidental trigger)
                duration = len(audio) / config.SAMPLE_RATE
                if duration < 0.3:
                    logger.info("⏩ Recording too short, ignoring.")
                    print("   (too short, ignoring)")
                    continue

                t_after_record = time.perf_counter()
                logger.info(f"⏱️  Recording took {t_after_record - t_start:.2f}s")

                # ── Step 4: Transcribe ──
                shared_state.current_status = "thinking"
                user_text = self.stt.transcribe(audio)
                
                if user_text:
                    shared_state.current_user_text = user_text

                t_after_stt = time.perf_counter()
                logger.info(f"⏱️  STT took {t_after_stt - t_after_record:.2f}s")

                if not user_text or len(user_text.strip()) < 2:
                    logger.info("⏩ No speech detected, returning to idle.")
                    print("   (couldn't hear anything, try again)")
                    shared_state.current_status = "idle"
                    continue

                print(f"   📝 You said: \"{user_text}\"")

                # ── Step 5: Fetch memory context (with semantic search) ──
                memory_context = self.memory.get_context_for_prompt(user_text)

                # ── Step 6: Stream LLM → TTS (concurrent) ──
                t_llm_start = time.perf_counter()
                print("   🤖 Thinking & speaking...")

                # Create LLM token stream
                token_stream = llm_module.stream_response(
                    user_text, memory_context
                )

                # Tee the stream: collect full response while TTS consumes it
                full_response_parts = []
                first_token_time = [None]  # mutable container

                def tee_generator():
                    for token in token_stream:
                        if first_token_time[0] is None:
                            first_token_time[0] = time.perf_counter()
                        full_response_parts.append(token)
                        yield token

                # TTS consumes the stream and speaks in real time
                self.tts.speak_stream(tee_generator())

                full_response = "".join(full_response_parts)
                t_end = time.perf_counter()

                # ── Step 7: Process self-edit memory tags (Letta Layer 5) ──
                # The LLM may have included [REMEMBER], [UPDATE], [FORGET] tags
                # Process them and get the cleaned response
                if full_response.strip():
                    cleaned_response = self.memory.process_self_edits(full_response)
                else:
                    cleaned_response = full_response

                # Timing breakdown
                total_time = t_end - t_start
                first_token_latency = (
                    first_token_time[0] - t_llm_start
                    if first_token_time[0]
                    else 0
                )

                logger.info(f"⏱️  First LLM token: {first_token_latency:.2f}s")
                logger.info(f"⏱️  Total interaction: {total_time:.2f}s")
                print(
                    f"   🤖 \"{cleaned_response[:100]}"
                    f"{'...' if len(cleaned_response) > 100 else ''}\""
                )
                print(
                    f"   ⏱️  Total: {total_time:.1f}s "
                    f"(first token: {first_token_latency:.1f}s)"
                )

                # ── Step 8: Store in all memory layers (background) ──
                if cleaned_response.strip():
                    self.memory.store_conversation(user_text, cleaned_response, full_response)

                # ── Back to idle ──
                logger.info("─" * 40)
                print("\n   👂 Listening for wake word...\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Pipeline error: {e}", exc_info=True)
                print(f"   ❌ Error: {e}")
                print("   Returning to idle...\n")
                continue

    def stop(self):
        """Gracefully shut down all engines."""
        logger.info("🛑 Shutting down assistant...")
        self._running = False
        self.tts.stop()
        self.wake_detector.cleanup()
        logger.info("👋 Assistant stopped.")
