# 🎙️ Local Voice Assistant

A fast, fully offline voice assistant powered by local AI models. Zero cloud APIs.

## Architectures

```
Ollama (Vulkan GPU) ← standard local deployment
        ↓
Wake Word (Porcupine) → 🔔 Beep → Record Speech → STT (faster-whisper) →
LLM (streaming chat) → TTS (Kokoro, concurrent) → Memory (Mem0 + Letta) → Idle
```

**Key design**: LLM streams tokens → TTS speaks sentence-by-sentence as they arrive → you hear the response *while* the LLM is still thinking.

## Prerequisites

- **Python 3.9+**
- **Ollama** installed on your system
- A working **microphone** and **speakers**
- **Windows** (tested on Windows 10/11)
- **AMD GPU with Vulkan support** (tested on RX 6500 XT)

## Installation

### 1. Set Up Ollama with Vulkan

1. Download and install **Ollama**.
2. Set the `OLLAMA_VULKAN=1` environment variable to enable Vulkan acceleration for your AMD GPU. Open PowerShell and run:
   ```bash
   setx OLLAMA_VULKAN 1
   ```
3. Restart your terminal (and fully quit/restart the Ollama app from the system tray) so the environment variable takes effect.
4. Pull the required models:
   ```bash
   ollama pull qwen3-fast
   ```

### 2. Create Virtual Environment

```bash
cd c:\Users\amana\Desktop\Antigravity\Antigravity_
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download TTS Model Files

**Option A — Automatic:**
```bash
python setup_models.py
```

**Option B — Manual:**
Download these two files into the project folder:
- [kokoro-v1.0.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx) (~300MB)
- [voices-v1.0.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin) (~5MB)

> **Note**: `faster-whisper` will download the Whisper `tiny` model (~75MB) on first run automatically.

### 5. (Optional) Set Up Mem0 Semantic Search

If you want semantic memory search (in addition to the reliable core memory), install Ollama and the embedding model:
```bash
pip install mem0ai
ollama pull nomic-embed-text
ollama serve
```
Core memory works without this — Mem0 is a bonus layer.

## Running

```bash
python main.py
```

The assistant will:
1. Connect to your local Ollama instance
2. Listen for the wake word

Say **"Porcupine"** to activate, speak your question, and the assistant will respond aloud.

## File Structure

```
├── main.py              # Entry point, logging, signal handling
├── assistant.py         # Pipeline orchestrator + llama-server lifecycle
├── config.py            # All settings in one place
├── wake_word.py         # Porcupine wake word detection
├── stt.py               # faster-whisper speech-to-text
├── llm.py               # llama-server streaming client (OpenAI-compatible)
├── tts.py               # Kokoro TTS with chunked playback
├── memory_manager.py    # Mem0 + Letta-inspired core memory
├── memory.py            # (Legacy) JSON memory tree
├── audio_utils.py       # Mic recording utilities
├── setup_models.py      # Downloads Kokoro model files
├── requirements.txt     # Python dependencies
├── core_memory.json     # Auto-created: persistent user facts (Letta-style)
├── memory.json          # (Legacy) old memory format
├── kokoro-v1.0.onnx     # TTS model (downloaded via setup_models.py)
├── voices-v1.0.bin      # TTS voices (downloaded via setup_models.py)
└── README.md            # This file
```

## Configuration

Edit `config.py` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `"qwen3-fast"` | The Ollama model to use for the assistant |
| `WHISPER_MODEL_SIZE` | `"tiny"` | STT model (`tiny`, `base`, `small`) |
| `KOKORO_VOICE` | `"af_heart"` | TTS voice |
| `KOKORO_SPEED` | `1.15` | Speech speed multiplier |
| `BEEP_FREQUENCY` | `800` | Wake word beep frequency (Hz) |
| `SILENCE_DURATION` | `1.5` | Seconds of silence to stop recording |
| `PORCUPINE_SENSITIVITIES` | `[0.7]` | Wake word sensitivity (0-1) |

### Vulkan Optimization

The RX 6500 XT has an AMD GPU. The system is configured to use Ollama with Vulkan enabled via the `OLLAMA_VULKAN=1` environment variable, ensuring fast token generation with AMD cards.

## Memory

The assistant remembers things about you across sessions using a two-layer system:

1. **Core Memory** (Letta-inspired) — Always reliable, stored in `core_memory.json`
   - **Persona block**: the assistant's personality and identity
   - **Human block**: accumulated facts about you, auto-updated each conversation
   - Tell it "My name is Alex" and it will remember forever

2. **Mem0** (optional) — Semantic vector search for relevant memories
   - Requires Ollama + nomic-embed-text
   - Adds semantic search ("What does my friend like?")
   - Core memory works without it

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Connection error during generation | Ensure **Ollama** is running in the background |
| Ollama not using GPU | Verify that `OLLAMA_VULKAN=1` is set in your system environment variables and Ollama was restarted |
| Porcupine init fails | Check API key in `config.py`, or use fallback mode |
| "Kokoro model not found" | Run `python setup_models.py` |
| No audio output | Check default speakers in Windows Sound settings |
| Mic not working | Check default mic in Windows Sound settings |
| Slow STT | Change `WHISPER_MODEL_SIZE` to `"tiny"` in config | 
