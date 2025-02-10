from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
from ultralytics import YOLO
import asyncio
import uvicorn
import numpy as np
import logging
from typing import Optional

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load YOLO model
model_path = 'models/best.pt'  # Configurable model path
model = YOLO(model_path)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_text("Error: Could not open camera")
        await websocket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera")
                break

            # Perform detection
            results = model(frame)
            annotated_frame = results[0].plot()  # Get annotated frame

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                logging.error("Failed to encode frame to JPEG")
                continue

            # Send frame through WebSocket
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.033)  # ~30fps

    except Exception as e:
        logging.error(f"Error in WebSocket connection: {e}")
    finally:
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastAPI YOLO WebSocket Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to the YOLO model")
    args = parser.parse_args()

    model_path = args.model
    model = YOLO(model_path)
  
    uvicorn.run(app, host=args.host, port=args.port)
