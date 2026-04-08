"""
shared_state.py — Global state shared between the Assistant pipeline and the Web Server.
"""

import time
import threading

# The current activity phase of the voice assistant
# States: "idle", "listening", "recording", "thinking", "speaking"
current_status = "idle"

# Real-time microphone volume level (normalized 0.0 to 1.0 roughly, or raw RMS mapped)
current_volume = 0.0

# The latest transcribed text from the user
current_user_text = ""

# The ongoing generated response text from the LLM
current_response_text = ""

# The currently selected personality layout key
current_personality = "unfiltered"

# The detected emotion of the user's latest input
# Values: "joyful", "excited", "positive", "neutral", "negative", "distressed", "sarcastic"
current_emotion = "neutral"

# Current mood energy level and response mode (from mood_tracker)
current_mood_energy = "moderate"
current_mood_mode = "engage"

# Timestamp of the last user interaction (for proactive idle detection)
last_interaction_time = time.time()

# Mutex to prevent proactive speech from colliding with active interactions
interaction_lock = threading.Lock()

