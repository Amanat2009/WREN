"""
tts.py — Text-to-Speech using Kokoro ONNX.
🚀 OPTIMIZED: Separate synthesis and playback threads run in parallel.
   Synthesis of chunk N+1 happens while chunk N is playing.
   Warmup on init. Smaller queue timeouts.

Requires model files in the project directory:
  - kokoro-v1.0.onnx
  - voices-v1.0.bin
Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
"""

import logging
import os
import queue
import re
import threading
import time
from typing import Generator

import numpy as np
import sounddevice as sd

import config
import shared_state

logger = logging.getLogger(__name__)

# Model file paths (same directory as this script)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(_BASE_DIR, "voices-v1.0.bin")


class TTSEngine:
    """
    Kokoro-based TTS engine with streaming support.

    Architecture (two background threads):
        Main thread: buffers LLM tokens → pushes text chunks to _text_queue
        Synthesis thread: reads _text_queue → synthesizes → pushes audio to _audio_queue
        Playback thread: reads _audio_queue → plays audio via sounddevice

    This means synthesis of the NEXT chunk overlaps with playback of the CURRENT chunk.
    """

    def __init__(self):
        logger.info("⏳ Loading Kokoro TTS model...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Kokoro model not found at {MODEL_PATH}\n"
                "Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0\n"
                "Or run: python setup_models.py"
            )
        if not os.path.exists(VOICES_PATH):
            raise FileNotFoundError(
                f"Kokoro voices not found at {VOICES_PATH}\n"
                "Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0\n"
                "Or run: python setup_models.py"
            )

        from kokoro_onnx import Kokoro

        self.kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
        self._text_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        logger.info("✅ Kokoro TTS loaded.")

    def warmup(self):
        """Run a dummy synthesis to warm up ONNX runtime."""
        logger.info("🔥 Warming up TTS...")
        t0 = time.perf_counter()
        try:
            self.kokoro.create(
                "Hello.",
                voice=config.KOKORO_VOICE,
                speed=config.KOKORO_SPEED,
                lang="en-us",
            )
            logger.info(f"✅ TTS warmed up ({time.perf_counter() - t0:.2f}s).")
        except Exception as e:
            logger.warning(f"⚠️  TTS warmup failed (non-fatal): {e}")

    def speak_stream(self, text_generator: Generator[str, None, None], interrupt_callback=None):
        """
        Consume an LLM token stream and speak it in real time.

        Three concurrent activities:
          1. THIS thread: buffers tokens → flushes text chunks to _text_queue
          2. Synthesis thread: _text_queue → Kokoro → _audio_queue
          3. Playback thread: _audio_queue → sounddevice
        """
        self._stop_event.clear()
        self._text_queue = queue.Queue()
        self._audio_queue = queue.Queue()

        # Start both worker threads
        synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        synth_thread.start()
        play_thread.start()

        buffer = ""
        chunk_count = 0

        for token in text_generator:
            buffer += token

            should_flush = False

            # Flush on sentence boundary (if buffer long enough)
            if len(buffer) >= config.TTS_MIN_CHUNK_LENGTH:
                for delimiter in config.TTS_SENTENCE_DELIMITERS:
                    if delimiter in buffer:
                        should_flush = True
                        break

            # Force flush if buffer getting too long
            if len(buffer) >= config.TTS_MAX_CHUNK_LENGTH:
                should_flush = True

            if should_flush:
                flush_text, remaining = self._split_at_boundary(buffer)
                cleaned = self._clean_text(flush_text)
                if cleaned:
                    self._text_queue.put(cleaned)
                    chunk_count += 1
                buffer = remaining

        # Flush remaining
        remaining_cleaned = self._clean_text(buffer)
        if remaining_cleaned:
            self._text_queue.put(remaining_cleaned)
            chunk_count += 1

        # Signal end
        self._text_queue.put(None)

        # Wait for both threads indefinitely or until interrupted
        while True:
            if not synth_thread.is_alive() and not play_thread.is_alive():
                break
                
            if interrupt_callback and interrupt_callback():
                logger.info("🛑 Barge-in detected! Intercepting TTS.")
                self.stop()
                break
                
            time.sleep(0.05)

        logger.info(f"🔊 TTS complete — {chunk_count} chunks spoken.")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip markdown formatting and memory tags so TTS doesn't pronounce them."""
        # Self-edit memory tags: [REMEMBER: ...], [UPDATE: ...], [UPDATE_RELATIONSHIP: ...], [FORGET: ...]
        text = re.sub(r'\[(REMEMBER|UPDATE|UPDATE_RELATIONSHIP|FORGET):\s*[^\]]*\]', '', text, flags=re.IGNORECASE)
        # Think tags (safety net)
        text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
        # Markdown formatting
        text = re.sub(r'\*+', '', text)      # asterisks (bold / italic)
        text = re.sub(r'_+', ' ', text)       # underscores
        text = re.sub(r'`+', '', text)        # backticks
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # headers
        text = re.sub(r'\s{2,}', ' ', text)   # collapse extra whitespace
        return text.strip()

    @staticmethod
    def _split_at_boundary(text: str) -> tuple:
        """Split text at the last sentence boundary. Returns (flush, remaining)."""
        for i in range(len(text) - 1, -1, -1):
            if text[i] in config.TTS_SENTENCE_DELIMITERS:
                return text[: i + 1], text[i + 1 :]
        return text, ""

    def _synth_worker(self):
        """Synthesis thread: reads text from queue, synthesizes audio, forwards to playback."""
        chunk_index = 0
        while not self._stop_event.is_set():
            try:
                text = self._text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text is None:
                # Forward end signal to playback
                self._audio_queue.put(None)
                break

            try:
                t0 = time.perf_counter()

                # ── Emotion-aware voice/speed modulation ──
                emotion = shared_state.current_emotion
                voice, speed = config.EMOTION_TTS_MAP.get(
                    emotion,
                    (config.KOKORO_VOICE, config.KOKORO_SPEED),
                )

                samples, sample_rate = self.kokoro.create(
                    text,
                    voice=voice,
                    speed=speed,
                    lang="en-us",
                )

                if not isinstance(samples, np.ndarray):
                    samples = np.array(samples, dtype=np.float32)

                elapsed = time.perf_counter() - t0
                logger.debug(
                    f"🗣️  Synth chunk {chunk_index} ({elapsed:.2f}s) "
                    f"[{emotion}→{voice}@{speed}x]: \"{text[:40]}...\""
                )
                self._audio_queue.put((samples, sample_rate))
                chunk_index += 1

            except Exception as e:
                logger.error(f"❌ TTS synthesis error on chunk {chunk_index}: {e}")

    def _playback_worker(self):
        """Playback thread: writes audio chunks into a single continuous stream (no gaps)."""
        stream = None
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    break

                samples, sample_rate = item

                # Ensure float32 for the output stream
                if samples.dtype != np.float32:
                    samples = samples.astype(np.float32)

                # Open the stream lazily on first chunk (so we know the sample rate)
                if stream is None:
                    stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                    )
                    stream.start()

                # Write directly into the continuous stream — no gaps between chunks
                try:
                    stream.write(samples.reshape(-1, 1))
                except sd.PortAudioError as pe:
                    logger.warning(f"⚠️ Audio stream error/underflow: {pe}. Recreating stream...")
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
                    stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                    )
                    stream.start()
                    stream.write(samples.reshape(-1, 1))

        except Exception as e:
            logger.error(f"❌ Audio playback worker crashed: {e}")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    def stop(self):
        """Stop any ongoing playback."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass

    def speak_text(self, text: str):
        """Speak a single text string (non-streaming, blocking)."""

        def _gen():
            yield text

        self.speak_stream(_gen())
