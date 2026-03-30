"""
audio_utils.py — Microphone recording with silence-based endpoint detection.
🚀 OPTIMIZED: pre-allocated buffer, fast RMS, trailing silence trimming.
"""

import logging
import numpy as np
import sounddevice as sd

import config
import shared_state

logger = logging.getLogger(__name__)


def _rms_fast(audio_chunk: np.ndarray) -> float:
    """Fast RMS: avoids float32 cast overhead by using int32 accumulation."""
    # int16 squared fits in int32 without overflow for chunks < 65k samples
    chunk_i32 = audio_chunk.astype(np.int32)
    return float(np.sqrt(np.mean(chunk_i32 * chunk_i32)))


def calibrate_ambient_noise(duration_sec: float = 1.0):
    """
    Measure the room's ambient noise floor and dynamically update SILENCE_THRESHOLD.
    Call this ONCE at startup to adapt to the current environment.
    """
    chunk_samples = int(config.SAMPLE_RATE * config.RECORD_CHUNK_MS / 1000)
    max_chunks = int(duration_sec * 1000 / config.RECORD_CHUNK_MS)
    
    logger.info(f"🎙️  Calibrating room noise for {duration_sec}s...")
    
    ambient_energies = []
    
    try:
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            blocksize=chunk_samples,
        ) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                energy = _rms_fast(chunk.flatten())
                ambient_energies.append(energy)
                
        if ambient_energies:
            avg_ambient = sum(ambient_energies) / len(ambient_energies)
            # Update the global config. Add the fixed threshold as a "margin" above the floor.
            config.SILENCE_THRESHOLD = avg_ambient + 350 # Original fixed constant was 350
            logger.info(f"🎧 Ambient ceiling: {avg_ambient:.0f} | Auto-Threshold: {config.SILENCE_THRESHOLD:.0f}")
    except Exception as e:
        logger.error(f"❌ Microphone calibration failed: {e}. Using default threshold.")


def record_speech() -> np.ndarray:
    """
    Record speech from the default microphone.

    Listens until either:
      - SILENCE_DURATION seconds of consecutive silence, or
      - RECORD_MAX_SECONDS total elapsed.

    Returns:
        numpy int16 array of recorded audio at SAMPLE_RATE Hz,
        with trailing silence trimmed.
    """
    chunk_samples = int(config.SAMPLE_RATE * config.RECORD_CHUNK_MS / 1000)
    max_chunks = int(config.RECORD_MAX_SECONDS * 1000 / config.RECORD_CHUNK_MS)
    silence_chunks_needed = int(config.SILENCE_DURATION * 1000 / config.RECORD_CHUNK_MS)

    logger.info("🎙️  Recording... (speak now)")

    buffer = np.empty(max_chunks * chunk_samples, dtype=np.int16)
    write_pos = 0
    silence_count = 0
    speech_detected = False
    last_speech_pos = 0

    try:
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            blocksize=chunk_samples,
        ) as stream:
            for i in range(max_chunks):
                try:
                    chunk, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        logger.warning("⚠️  Audio overflow detected.")
                    chunk = chunk.flatten()
                except Exception as e:
                    logger.error(f"❌ Microphone read failed: {e}")
                    break

                # Write directly into pre-allocated buffer
                end_pos = write_pos + len(chunk)
                buffer[write_pos:end_pos] = chunk
                write_pos = end_pos

                energy = _rms_fast(chunk)

                # Normalize for visualizer
                shared_state.current_volume = min(1.0, energy / 4000.0)

                if energy > config.SILENCE_THRESHOLD:
                    silence_count = 0
                    if not speech_detected:
                        logger.info("🗣️  Speech detected.")
                        speech_detected = True
                    last_speech_pos = write_pos
                else:
                    silence_count += 1

                # Only stop on silence AFTER we've heard some speech
                if speech_detected and silence_count >= silence_chunks_needed:
                    logger.info("🔇  Silence detected — stopping recording.")
                    break
    except Exception as e:
        logger.error(f"❌ Failed to open microphone: {e}")
        return np.array([], dtype=np.int16)

    # Trim trailing silence
    trim_margin = int(config.SAMPLE_RATE * 0.5)
    trim_pos = min(write_pos, last_speech_pos + trim_margin)
    if trim_pos < chunk_samples:
        trim_pos = write_pos

    shared_state.current_volume = 0.0

    audio = buffer[:trim_pos]
    duration = len(audio) / config.SAMPLE_RATE
    logger.info(f"📼  Recorded {duration:.1f}s of audio ({len(audio)} samples)")
    return audio


def play_audio(audio: np.ndarray, sample_rate: int = config.TTS_SAMPLE_RATE):
    """Play a numpy audio array through the default speakers (blocking)."""
    try:
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        logger.error(f"❌ Playback failed: {e}")
