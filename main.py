import argparse
from ultralytics import YOLO
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    """Endpoint to check if the YOLO web server is running."""
    return {"message": "YOLO Web Server Running"}

def run_web_server():
    """Starts a FastAPI web server on port 8000."""
    print("Starting web server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

def train_model():
    """Trains the YOLO model using the specified dataset and parameters."""
    print("Training YOLO model...")
    model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model
    model.train(data="data.yaml", epochs=50, imgsz=640, batch=8)
    model.export(format="onnx")  # Convert for deployment

def evaluate_model():
    """Loads and evaluates the YOLO model on test images and webcam feed."""
    print("Evaluating YOLO model...")
    model = YOLO("yolov8n.pt")
    model.predict(source="dataset/images/test", show=True, conf=0.5)
    model.predict(source=0, show=True, conf=0.5)

def main():
    """Parses command-line arguments and runs the corresponding function."""
    parser = argparse.ArgumentParser(description="YOLO Command-line tool")
    parser.add_argument("-w", action="store_true", help="Run web server")
    parser.add_argument("-t", action="store_true", help="Train the model")
    parser.add_argument("-e", action="store_true", help="Evaluate the model")
    
    args = parser.parse_args()
    
    if args.w:
        run_web_server()
    elif args.t:
        train_model()
    elif args.e:
        evaluate_model()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
