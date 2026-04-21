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
from mood_tracker import MoodTracker
from proactive import ProactiveEngine
import audio_utils
import llm as llm_module
import config
import shared_state
import web_server
import emotion

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

        # Mood tracker (persistent emotional intelligence)
        self.mood = MoodTracker()

        # ⚡ Warmup all engines in parallel to reduce startup time
        warmup_threads = [
            threading.Thread(target=self.stt.warmup, daemon=True),
            threading.Thread(target=self.tts.warmup, daemon=True),
            threading.Thread(target=llm_module.warmup, daemon=True),
            threading.Thread(target=emotion.warmup, daemon=True),
        ]
        for t in warmup_threads:
            t.start()
        for t in warmup_threads:
            t.join(timeout=30)

        # 🌐 Start the Web UI Server
        web_server.start_server_in_background()

        # 🎧 Calibrate the microphone to the room's ambient noise floor
        audio_utils.calibrate_ambient_noise(duration_sec=1.0)

        # 💬 Proactive engagement engine (startup greeting, idle check-ins)
        self.proactive = ProactiveEngine(self.tts, self.memory, self.mood)

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
        self.is_continuous_mode = False
        import queue
        logger.info("🎧 Voice assistant is READY. Say the wake word to begin!")
        print("\n" + "=" * 50)
        print("  🎙️  LOCAL VOICE ASSISTANT")
        print("  Say 'Porcupine' to activate")
        print("  Press Ctrl+C to quit")
        print("=" * 50 + "\n")

        # Start proactive engagement (greeting thread fires now)
        self.proactive.start()

        while self._running:
            try:
                # ── Reset state for new interaction ──
                is_continuous = self.is_continuous_mode
                self.is_continuous_mode = False

                if is_continuous:
                    shared_state.current_status = "listening"
                    detected = True
                else:
                    shared_state.current_status = "idle"
                    detected = self.wake_detector.wait_for_wake_word()

                if not detected or not self._running:
                    continue

                # Acquire interaction lock so proactive engine doesn't collide
                shared_state.interaction_lock.acquire()

                # Clear text state after wake word is detected to preserve subtitles
                shared_state.current_user_text = ""
                shared_state.current_response_text = ""

                # ── Step 2: BEEP + Acknowledge wake ──
                if not is_continuous:
                    shared_state.current_status = "listening"
                    self._play_beep()
                    print("\n🔔 Wake word detected! Listening...")
                    logger.info("─" * 40)
                else:
                    print("\n👂 Listening for reply...")
                    logger.info("─" * 40)

                t_start = time.perf_counter()

                # ── Step 3: Record user speech (with Streaming STT) ──
                shared_state.current_status = "recording"
                
                audio_queue = queue.Queue()
                
                def _stt_streaming_worker():
                    while True:
                        audio_data = audio_queue.get()
                        if audio_data is None: 
                            break
                        try:
                            text = self.stt.transcribe(audio_data, is_final=False)
                            if text:
                                shared_state.current_user_text = text
                        except Exception:
                            pass
                            
                stt_worker_thread = threading.Thread(target=_stt_streaming_worker, daemon=True)
                stt_worker_thread.start()

                def _on_audio(audio_data):
                    while not audio_queue.empty():
                        try: audio_queue.get_nowait()
                        except Exception: pass
                    audio_queue.put(audio_data)

                timeout_val = config.CONTINUOUS_LISTEN_TIMEOUT if is_continuous else 0.0
                audio = audio_utils.record_speech(timeout_sec=timeout_val, chunk_callback=_on_audio)
                
                audio_queue.put(None)
                stt_worker_thread.join()

                # If timeout reached and no audio recorded
                if len(audio) == 0:
                    logger.info("⏩ Continuous listen timed out. Going back to idle.")
                    shared_state.interaction_lock.release()
                    continue

                # Skip if recording is too short (noise/accidental trigger)
                duration = len(audio) / config.SAMPLE_RATE
                if duration < 0.3:
                    logger.info("⏩ Recording too short, ignoring.")
                    print("   (too short, ignoring)")
                    shared_state.interaction_lock.release()
                    continue

                t_after_record = time.perf_counter()
                logger.info(f"⏱️  Recording took {t_after_record - t_start:.2f}s")

                # ── Step 4: Transcribe (Final) ──
                shared_state.current_status = "thinking"
                user_text = self.stt.transcribe(audio, is_final=True)
                
                if user_text:
                    shared_state.current_user_text = user_text

                t_after_stt = time.perf_counter()
                logger.info(f"⏱️  STT took {t_after_stt - t_after_record:.2f}s")

                if not user_text or len(user_text.strip()) < 2:
                    logger.info("⏩ No speech detected, returning to idle.")
                    print("   (couldn't hear anything, try again)")
                    shared_state.current_status = "idle"
                    shared_state.interaction_lock.release()
                    continue

                print(f"   📝 You said: \"{user_text}\"")

                # ── Step 5a: Emotion analysis (~16ms, synchronous) ──
                emo = emotion.analyze(user_text)
                shared_state.current_emotion = emo.category
                logger.info(
                    f"🎭 Emotion: {emo.category} "
                    f"(compound={emo.compound:.3f}, humor={emo.is_humor})"
                )
                print(f"   🎭 Emotion: {emo.category}"
                      f"{' 😄' if emo.is_humor else ''}")

                # ── Step 5b: Log mood + analyze trend (<1ms) ──
                self.mood.log(emo, user_text)
                mood = self.mood.analyze(emo, user_text)
                logger.info(
                    f"📊 Mood: trend={mood.trend}, energy={mood.energy}, "
                    f"concern={mood.concern}, mode={mood.response_mode}"
                )

                # ── Step 5c: Fetch memory context (with semantic search) ──
                memory_context = self.memory.get_context_for_prompt(user_text)

                # ── Step 5d: Build emotional intelligence context ──
                # Layer 1: Base emotion modifier
                emotion_modifier = config.EMOTION_PROMPT_MODIFIERS.get(
                    emo.category, ""
                )
                if emo.is_humor:
                    emotion_modifier = (
                        emotion_modifier + " " + config.HUMOR_PROMPT_MODIFIER
                    ).strip()

                # Layer 2: Energy matching
                energy_modifier = config.ENERGY_PROMPT_MODIFIERS.get(
                    mood.energy, ""
                )

                # Layer 3: Response mode (validation layer)
                mode_modifier = config.RESPONSE_MODE_MODIFIERS.get(
                    mood.response_mode, ""
                )

                # Layer 4: Mood trend summary
                mood_summary = mood.summary

                # Assemble full emotional context
                emo_parts = [
                    f"Detected mood: {emo.category}"
                    f"{' (with humor)' if emo.is_humor else ''}",
                    f"Energy level: {mood.energy}",
                    f"Mood trend: {mood.trend}",
                    f"Response posture: {mood.response_mode}",
                ]
                if emotion_modifier:
                    emo_parts.append(emotion_modifier)
                if energy_modifier:
                    emo_parts.append(energy_modifier)
                if mode_modifier:
                    emo_parts.append(mode_modifier)
                if mood_summary:
                    emo_parts.append(mood_summary)

                memory_context += (
                    f"\n\n=== EMOTIONAL INTELLIGENCE ===\n"
                    + "\n".join(emo_parts)
                )

                # ── Step 6: Stream LLM → TTS (concurrent) ──
                t_llm_start = time.perf_counter()
                
                search_iterations = 0
                while search_iterations < 2:
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
                            
                    # Monitor for barge-in while TTS plays
                    barge_in_event = threading.Event()
                    tts_done_event = threading.Event()
                    
                    def _monitor_barge_in():
                        if self.wake_detector.wait_for_wake_word(stop_event=tts_done_event):
                            barge_in_event.set()
                            
                    barge_thread = threading.Thread(target=_monitor_barge_in, daemon=True)
                    barge_thread.start()

                    def _check_barge_in():
                        return barge_in_event.is_set()

                    # TTS consumes the stream and speaks in real time
                    self.tts.speak_stream(tee_generator(), interrupt_callback=_check_barge_in)
                    
                    tts_done_event.set() # Stop the monitor when TTS finishes normally

                    full_response = "".join(full_response_parts)
                    t_end = time.perf_counter()
                    
                    if barge_in_event.is_set():
                        print("\n🛑 Barge-in detected! Stopping TTS...")
                        self.is_continuous_mode = True # Instantly loop to listening
                        break

                    # ── Step 7: Process self-edit memory tags (Letta Layer 5) ──
                    if full_response.strip():
                        cleaned_response = self.memory.process_self_edits(full_response)
                        
                        # Process SEARCH tag
                        import re
                        search_match = re.search(r'\[SEARCH:\s*([^\]]+)\]', full_response, re.IGNORECASE)
                        if search_match:
                            query = search_match.group(1).strip()
                            print(f"   🔍 Web Search Triggered: {query}")
                            try:
                                from tavily import TavilyClient
                                tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
                                search_result = tavily_client.search(
                                    query=query, 
                                    max_results=config.TAVILY_MAX_RESULTS,
                                    search_depth=config.TAVILY_SEARCH_DEPTH
                                )
                                context_strs = []
                                for res in search_result.get('results', []):
                                    context_strs.append(f"- {res.get('title', '')}: {res.get('content', '')}")
                                context = "\n".join(context_strs)
                                memory_context += f"\n\n=== WEB SEARCH RESULTS for '{query}' ===\n{context}"
                                search_iterations += 1
                                print("   🌐 Search complete. Re-prompting LLM...")
                                continue
                            except Exception as e:
                                print(f"   ⚠️ Search failed: {e}")
                                memory_context += f"\n\n=== WEB SEARCH RESULTS for '{query}' ===\nSearch failed: {e}"
                                search_iterations += 1
                                continue
                    else:
                        cleaned_response = full_response
                        
                    break # Break out of the search loop if no jump is required

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

                # ── Step 8: Detect GF emotional state from her reply ──
                # Runs <1ms (regex only). Updates current_gf_emotion so
                # the TTS synth worker picks the right voice/speed profile.
                if cleaned_response.strip():
                    import emotion as emotion_module
                    gf_emo = emotion_module.analyze_response(cleaned_response)
                    shared_state.current_gf_emotion = gf_emo
                    print(f"   💕 GF Emotion: {gf_emo}")

                # ── Step 8b: Update relationship warmth score ──
                # Nudges warmth based on user's emotional valence this turn
                _warmth_deltas = {
                    "excited":    +0.020,
                    "joyful":     +0.015,
                    "positive":   +0.008,
                    "neutral":    +0.003,
                    "sarcastic":  +0.002,
                    "negative":   -0.010,
                    "distressed": -0.020,
                }
                _warmth_delta = _warmth_deltas.get(emo.category, 0.0)
                try:
                    self.memory.core.update_warmth(_warmth_delta)
                except Exception:
                    pass

                # ── Step 9: Store in all memory layers (background) ──
                if cleaned_response.strip():
                    self.memory.store_conversation(user_text, cleaned_response, full_response)

                # Update last interaction time for proactive engine
                shared_state.last_interaction_time = time.time()

                # Release the interaction lock
                shared_state.interaction_lock.release()

                if barge_in_event.is_set():
                    continue

                # Set flag for continuous conversation
                self.is_continuous_mode = True

                # ── Back to idle ──
                logger.info("─" * 40)
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Make sure we release the lock on error
                if shared_state.interaction_lock.locked():
                    try:
                        shared_state.interaction_lock.release()
                    except RuntimeError:
                        pass
                logger.error(f"❌ Pipeline error: {e}", exc_info=True)
                print(f"   ❌ Error: {e}")
                print("   Returning to idle...\n")
                continue

    def stop(self):
        """Gracefully shut down all engines."""
        logger.info("🛑 Shutting down assistant...")
        self._running = False
        # Save her last emotional state for carryover to next session
        try:
            self.memory.core.save_session_mood(
                shared_state.current_gf_emotion,
                0.0,   # valence placeholder (mood_tracker holds the real value)
            )
        except Exception:
            pass
        self.proactive.stop()
        self.tts.stop()
        self.wake_detector.cleanup()
        logger.info("👋 Assistant stopped.")
