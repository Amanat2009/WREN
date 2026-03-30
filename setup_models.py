"""
setup_models.py — Downloads required model files for the voice assistant.
Run this once before first use: python setup_models.py
"""

import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    {
        "name": "kokoro-v1.0.onnx",
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "size_mb": 300,
    },
    {
        "name": "voices-v1.0.bin",
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        "size_mb": 5,
    },
]


def download_file(url: str, dest: str, name: str):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"  ✅ {name} already exists, skipping.")
        return

    print(f"  ⏳ Downloading {name}...")
    print(f"     {url}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r     {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print(f"\n  ✅ {name} downloaded.")


def main():
    print("=" * 50)
    print("  🔧 Voice Assistant — Model Setup")
    print("=" * 50)
    print()

    for model in MODELS:
        dest = os.path.join(BASE_DIR, model["name"])
        download_file(model["url"], dest, model["name"])

    print()
    print("✅ All models ready! Run: python main.py")
    print()


if __name__ == "__main__":
    main()
