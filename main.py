from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
from ultralytics import YOLO
import asyncio
import uvicorn
import numpy as np

app = FastAPI()

# Load YOLO model
model = YOLO('models/best.pt')  # Make sure to have your best.pt model in the directory

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
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = model(frame)
            annotated_frame = results[0].plot()  # Get annotated frame

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            # Send frame through WebSocket
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.033)  # ~30fps

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)