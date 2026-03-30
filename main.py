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
    pass


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
