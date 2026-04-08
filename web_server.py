"""
web_server.py — FastAPI server for the Voice Assistant 3D Web UI.
Serves static frontend files and provides a WebSocket for real-time state sync.
"""

import os
import time
import asyncio
import logging
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

import config
import shared_state

logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Assistant Web UI")

# Mount the frontend directory for static files (CSS, JS)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if not os.path.exists(FRONTEND_DIR):
    os.makedirs(FRONTEND_DIR)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def root():
    """Serve the main index.html file."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend not found. Please create frontend/index.html"}

class PersonalityRequest(BaseModel):
    personality: str

@app.post("/api/personality")
async def set_personality(req: PersonalityRequest):
    """API endpoint to change the assistant's personality."""
    if req.personality in config.LLM_PERSONALITIES:
        shared_state.current_personality = req.personality
        logger.info(f"🎭 Personality changed via Web UI to: {req.personality}")
        return {"status": "success", "personality": req.personality}
    return {"status": "error", "message": "Invalid personality key"}, 400

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that streams the assistant's state to the web UI at 30 fps.
    """
    await websocket.accept()
    logger.info("🌐 Web Client Connected to State Stream")
    
    try:
        while True:
            # 🏎️ Efficiency: Only include long text strings if they've updated
            user_text = shared_state.current_user_text
            response_text = shared_state.current_response_text
            
            payload = {
                "status": shared_state.current_status,
                "volume": shared_state.current_volume,
                "personality": shared_state.current_personality,
                "emotion": shared_state.current_emotion,
                "mood_energy": shared_state.current_mood_energy,
                "mood_mode": shared_state.current_mood_mode,
                "user_text": user_text,
                "response_text": response_text
            }

            await websocket.send_json(payload)
            # Sleep for ~33ms to achieve ~30fps update rate
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        logger.info("🌐 Web Client Disconnected")
    except Exception as e:
        logger.error(f"🌐 WebSocket Error: {e}")

def run_server():
    """Run the Uvicorn server blocking (used in the background thread)."""
    # Suppress uvicorn access logs to prevent console spam
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.disabled = True
    
    uvicorn.run(
        app, 
        host=config.WEB_HOST, 
        port=config.WEB_PORT,
        log_level="warning" # Only log warnings/errors from the server itself
    )

def start_server_in_background():
    """Start the FastAPI server in a daemon thread so it runs alongside the assistant."""
    logger.info(f"🌐 Starting Web UI Server on http://{config.WEB_HOST}:{config.WEB_PORT}")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
