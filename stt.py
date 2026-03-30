"""
stt.py — Speech-to-Text using faster-whisper.
🚀 OPTIMIZED: warmup on init, disabled VAD (we already endpoint),
   multi-threaded, condition_on_previous_text=False.
"""

import logging
import time
import numpy as np
from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


class SpeechToText:
    """Faster-whisper based speech recognition engine."""

    def __init__(self):
        logger.info(
            f"⏳ Loading Whisper model '{config.WHISPER_MODEL_SIZE}' "
            f"(device={config.WHISPER_DEVICE}, compute={config.WHISPER_COMPUTE_TYPE})..."
        )
        self.model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            cpu_threads=config.WHISPER_THREADS,
        )
        logger.info("✅ Whisper model loaded.")

    def warmup(self):
        """Run a dummy transcription to warm up the model (JIT, caches, etc.)."""
        logger.info("🔥 Warming up Whisper...")
        dummy = np.zeros(config.SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
        segments, _ = self.model.transcribe(
            dummy, beam_size=1, language=config.WHISPER_LANGUAGE
        )
        # Drain the generator
        for _ in segments:
            pass
        logger.info("✅ Whisper warmed up.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a numpy int16 audio array to text.

        Args:
            audio: numpy int16 array at SAMPLE_RATE Hz.

        Returns:
            Transcribed text string (stripped).
        """
        t0 = time.perf_counter()

        # faster-whisper expects float32 in [-1, 1]
        audio_float = audio.astype(np.float32) / 32768.0

        segments, info = self.model.transcribe(
            audio_float,
            beam_size=config.WHISPER_BEAM_SIZE,
            language=config.WHISPER_LANGUAGE,
            # 🎯 Enable VAD to filter silence/noise segments for cleaner results
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,   # Don't split on short pauses
                speech_pad_ms=200,             # Pad speech edges to avoid clipping
            ),
            # 🎯 Suppress hallucinations on near-silent audio
            no_speech_threshold=0.6,
            # ⚡ Don't condition on previous — avoids hallucination loops
            condition_on_previous_text=False,
            # ⚡ No word timestamps needed
            word_timestamps=False,
        )

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts).strip()
        elapsed = time.perf_counter() - t0
        logger.info(f"📝 Transcription ({elapsed:.2f}s): \"{full_text}\"")
        return full_text
