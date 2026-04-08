"""
llm.py — Ollama streaming LLM client.
🚀 OPTIMIZED: persistent Session for connection pooling, raw byte
   iteration for minimal parsing overhead, keep-alive enabled.
"""

import json
import logging
from typing import Generator

import requests

import config
import shared_state

logger = logging.getLogger(__name__)

# ⚡ Persistent session — reuses TCP connection across requests
_session = requests.Session()
_session.headers.update({"Connection": "keep-alive"})


def warmup():
    """Prime the Ollama connection and model loading with a tiny request."""
    logger.info(f"🔥 Warming up LLM ({config.OLLAMA_MODEL} via Ollama)...")
    try:
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": "Hi",
            "system": "Reply with one word.",
            "stream": False,
            "keep_alive": -1,
            "options": {"num_predict": 2},
        }
        resp = _session.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        logger.info("✅ LLM warmed up.")
    except Exception as e:
        logger.warning(f"⚠️  LLM warmup failed (non-fatal): {e}")


def stream_response(
    user_message: str,
    memory_context: str = "",
    update_ui: bool = True,
) -> Generator[str, None, None]:
    """
    Send a prompt to Ollama and yield response text chunks as they stream in.

    Args:
        user_message: The user's transcribed speech.
        memory_context: Optional memory summary to inject into the system prompt.

    Yields:
        Text chunks (partial words/sentences) as they arrive from the LLM.
    """
    from datetime import datetime

    # Build system prompt dynamically based on Web UI selection
    personality_key = shared_state.current_personality
    system_prompt = config.LLM_PERSONALITIES.get(
        personality_key, 
        config.LLM_PERSONALITIES["unfiltered"]
    )
    
    if memory_context:
        system_prompt += (
            f"\n\nHere is your memory and context about your friend:\n"
            f"{memory_context}"
        )

    # Inject real-time awareness (date, time, day of week)
    now = datetime.now()
    system_prompt += (
        f"\n\n=== CURRENT TIME ===\n"
        f"It is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
    )

    # Add self-edit memory instructions (Letta Layer 5)
    if update_ui:  # Only for user-facing responses, not internal calls
        system_prompt += config.SELF_EDIT_INSTRUCTIONS

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": user_message,
        "system": system_prompt,
        "stream": True,
        "keep_alive": -1,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "num_predict": config.LLM_MAX_TOKENS,
        },
    }

    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    logger.info(f"🤖 Sending to LLM: \"{user_message[:80]}\"")

    try:
        # ⚡ Use persistent session, stream with small chunk size
        resp = _session.post(
            url,
            json=payload,
            stream=True,
            timeout=(5, config.OLLAMA_TIMEOUT),  # ⚡ (connect_timeout, read_timeout)
        )
        resp.raise_for_status()

        if update_ui:
            shared_state.current_status = "thinking"
        first_token_yielded = False
        _in_think_block = False       # Track <think>...</think> sections
        _tag_buffer = ""              # Buffer to detect partial tags

        # ⚡ iter_lines with no decode overhead — we do it manually
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            token = data.get("response", "")
            if token:
                # ── Filter out <think>...</think> blocks from Qwen3 ──
                _tag_buffer += token

                # Process complete tags in the buffer
                while _tag_buffer:
                    if _in_think_block:
                        # Look for closing </think>
                        end_idx = _tag_buffer.find("</think>")
                        if end_idx != -1:
                            _in_think_block = False
                            _tag_buffer = _tag_buffer[end_idx + 8:]  # Skip past </think>
                        else:
                            # Still inside think block — might have partial </think>
                            # Keep last 8 chars in case </think> is split across tokens
                            if len(_tag_buffer) > 8:
                                _tag_buffer = _tag_buffer[-8:]
                            break
                    else:
                        # Look for opening <think>
                        start_idx = _tag_buffer.find("<think>")
                        if start_idx != -1:
                            # Yield everything before the tag
                            before = _tag_buffer[:start_idx]
                            if before:
                                if not first_token_yielded:
                                    if update_ui:
                                        shared_state.current_status = "speaking"
                                    first_token_yielded = True
                                if update_ui:
                                    shared_state.current_response_text += before
                                yield before
                            _in_think_block = True
                            _tag_buffer = _tag_buffer[start_idx + 7:]  # Skip past <think>
                        else:
                            # No tag found — check for partial "<" at end
                            # Keep potential partial tag start in buffer
                            safe_end = len(_tag_buffer)
                            for i in range(max(0, len(_tag_buffer) - 7), len(_tag_buffer)):
                                if _tag_buffer[i] == "<" and _tag_buffer[i:] in "<think>"[:len(_tag_buffer) - i]:
                                    safe_end = i
                                    break
                            
                            to_yield = _tag_buffer[:safe_end]
                            _tag_buffer = _tag_buffer[safe_end:]
                            
                            if to_yield:
                                if not first_token_yielded:
                                    if update_ui:
                                        shared_state.current_status = "speaking"
                                    first_token_yielded = True
                                if update_ui:
                                    shared_state.current_response_text += to_yield
                                yield to_yield
                            break

            if data.get("done", False):
                # Flush any remaining buffer (not inside think block)
                if _tag_buffer and not _in_think_block:
                    if update_ui:
                        shared_state.current_response_text += _tag_buffer
                    yield _tag_buffer
                break

        resp.close()

    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to Ollama. Is it running?")
        yield "Sorry, I can't reach my brain right now. Make sure Ollama is running."
    except requests.exceptions.Timeout:
        logger.error("❌ Ollama request timed out.")
        yield "Sorry, I'm taking too long to think. Try again."
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        yield f"Sorry, something went wrong: {e}"
