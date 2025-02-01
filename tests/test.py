import cv2
from ultralytics import YOLO

def run_yolo_realtime():
    # Load the YOLOv8 model
    model = YOLO('yolo11n.pt')  # Use 'yolov8s.pt', 'yolov8m.pt', etc. for different sizes
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 = default camera, change number if you have multiple cameras
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        # Read frame from camera
        success, frame = cap.read()
        
        if not success:
            print("Error: Could not read frame.")
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Process results
        for result in results:
            # Extract bounding boxes
            for box in result.boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Get class ID
                class_id = int(box.cls[0])
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label with class name and confidence
                label = f"{result.names[class_id]} {confidence:.2f}"
                
                # Put text label
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the processed frame
        cv2.imshow('YOLOv11 Real-Time Detection', frame)
        
        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo_realtime()