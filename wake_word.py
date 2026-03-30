"""
wake_word.py — Porcupine-based wake word detection.
Listens continuously on the microphone and returns when the wake word is heard.
Falls back to a simple energy-spike detector if Porcupine fails to initialize.
"""

import logging
import struct
import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects the wake word using Picovoice Porcupine."""

    def __init__(self):
        self._porcupine = None
        self._use_fallback = False
        self._init_porcupine()

    def _init_porcupine(self):
        """Try to initialize Porcupine; fall back to energy detection on failure."""
        try:
            import pvporcupine
            self._porcupine = pvporcupine.create(
                access_key=config.PORCUPINE_ACCESS_KEY,
                keywords=config.PORCUPINE_KEYWORDS,
                sensitivities=config.PORCUPINE_SENSITIVITIES,
            )
            logger.info(
                f"✅ Porcupine initialized — listening for: {config.PORCUPINE_KEYWORDS}"
            )
            logger.info(
                f"   Frame length: {self._porcupine.frame_length} samples @ "
                f"{self._porcupine.sample_rate} Hz"
            )
        except Exception as e:
            logger.warning(f"⚠️  Porcupine init failed: {e}")
            logger.warning("   Falling back to energy-based wake detection (clap/loud sound).")
            self._use_fallback = True

    def wait_for_wake_word(self) -> bool:
        """
        Block until the wake word (or fallback trigger) is detected.
        Returns True when triggered, False if shut down.
        """
        if self._use_fallback:
            return self._fallback_detect()
        return self._porcupine_detect()

    def _porcupine_detect(self) -> bool:
        """Continuous Porcupine detection loop."""
        frame_length = self._porcupine.frame_length
        sample_rate = self._porcupine.sample_rate

        logger.info("👂 Listening for wake word...")

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
        ) as stream:
            while True:
                data, _ = stream.read(frame_length)
                pcm = data.flatten()

                # Porcupine C-bindings strictly expect a native Python list
                pcm_list = pcm.tolist()

                keyword_index = self._porcupine.process(pcm_list)
                if keyword_index >= 0:
                    logger.info(
                        f"🔔 Wake word detected: '{config.PORCUPINE_KEYWORDS[keyword_index]}'"
                    )
                    return True

    def _fallback_detect(self) -> bool:
        """Simple energy-spike fallback: detects a loud sound (clap, etc.)."""
        chunk_samples = int(config.SAMPLE_RATE * config.RECORD_CHUNK_MS / 1000)
        threshold = config.SILENCE_THRESHOLD * 5  # Much louder than speech silence

        logger.info("👂 Listening for loud trigger sound (fallback mode)...")

        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
        ) as stream:
            while True:
                data, _ = stream.read(chunk_samples)
                energy = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                if energy > threshold:
                    logger.info(f"🔔 Fallback trigger detected (energy={energy:.0f})")
                    return True

    def cleanup(self):
        """Release Porcupine resources."""
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
            logger.info("🧹 Porcupine resources released.")
