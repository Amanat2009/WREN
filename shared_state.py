"""
shared_state.py — Global state shared between the Assistant pipeline and the Web Server.
"""

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
