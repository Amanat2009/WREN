"""
main.py — Entry point for the local voice assistant.
Configures logging, handles Ctrl+C, and runs the assistant.
"""

import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import config
from assistant import Assistant


def bootstrap_services():
    """Starts background services if not already running."""
    # Letta REST server is no longer fundamentally required as memory 
    # is handled locally by the custom memory_manager.py module.
    # Mem0 background setup is handled automatically in memory_manager.
    # Neo4j is configured for Aura (cloud), so no local startup is needed.
    
    print("⏳ Checking background services...")
    
    # Check and start Ollama
    ollama_url = f"{config.OLLAMA_BASE_URL}/"
    try:
        req = urllib.request.Request(ollama_url)
        with urllib.request.urlopen(req, timeout=1):
            pass
        print("✅ Ollama is already running.")
    except Exception:
        print("⏳ Ollama not found. Starting Ollama server in the background...")
        try:
            creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000) if sys.platform == "win32" else 0
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags
            )
            time.sleep(3)  # Give it a few seconds to bind port
            print("✅ Ollama started successfully.")
        except FileNotFoundError:
            print("❌ 'ollama' command not found. Please ensure Ollama is installed and in your PATH.")



def setup_logging():
    """Configure logging with the format from config."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy loggers
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Local Voice Assistant...")

    assistant = Assistant()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n👋 Goodbye!")
        assistant.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        assistant.stop()


if __name__ == "__main__":
    bootstrap_services()
    main()
