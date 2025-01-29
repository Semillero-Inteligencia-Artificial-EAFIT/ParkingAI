# ParkingAI (in process)
Parking AI system for Eafit univercity

Here's a **linear pipeline** for training and deploying a **YOLO model for parking slot detection**:

---

### **1️ Collect & Prepare Data**
   - Download a dataset or create it, (We can mix both thecniques for learning prouses)
   [PKLot](https://public.roboflow.com/object-detection/pklot)
   [CNRPark-EXT](http://cnrpark.it/)
   [Kaggle Parking Dataset](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset)
   [Kaggle Parking Dataset2](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset)
   - If needed, manually annotate images using **LabelImg** or **Roboflow**.
   - Ensure data is in **YOLO format** (bounding box labels in `.txt` files).

---

### **2️ Organize Dataset**
   - Structure dataset:
     ```
     /dataset
     ├── images
     │   ├── train  (Training images)
     │   ├── val    (Validation images)
     │   ├── test   (Test images)
     ├── labels
     │   ├── train  (Bounding boxes for training)
     │   ├── val    (Bounding boxes for validation)
     │   ├── test   (Bounding boxes for test)
     ├── data.yaml
     ```
   - Create `data.yaml`:
     ```yaml
     train: dataset/images/train
     val: dataset/images/val
     test: dataset/images/test
     nc: 1  # Number of classes
     names: ['parking_slot']
     ```

---

### **3️ Train YOLO Model**
   - Install dependencies:
     ```bash
     pip install ultralytics
     ```
   - Train the model:
     ```python
     from ultralytics import YOLO
     model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model
     model.train(data="data.yaml", epochs=50, imgsz=640, batch=8)
     ```
   - Save the trained model:
     ```python
     model.export(format="onnx")  # Convert for deployment
     ```

---

### **4️ Validate & Test**
   - Run inference on a test image:
     ```python
     results = model("test.jpg", save=True, conf=0.5)
     ```
   - Visualize predictions:
     ```python
     model.predict(source="dataset/images/test", show=True, conf=0.5)
     ```

---

### **5️ Deploy for Real-Time Detection**
   - Run real-time detection using a webcam:
     ```python
     model.predict(source=0, show=True, conf=0.5)
     ```
   - Integrate with **FastAPI** for a web service.

---

### **6 Final Deployment & Monitoring**
   - Deploy to **Raspberry Pi**




### References

[https://www.youtube.com/watch?v=VZXdkOo3xNI](https://www.youtube.com/watch?v=VZXdkOo3xNI)

[https://www.youtube.com/watch?v=F-884J2mnOY](https://www.youtube.com/watch?v=F-884J2mnOY)

[PKLot-detection](https://github.com/Mu3llr/PKLot-detection)

[Profesional proyect](https://github.com/DeepParking/DeepParking)

**no so imortant, but elegant**

[https://www.youtube.com/watch?v=MeSeuzBhq2E](https://www.youtube.com/watch?v=MeSeuzBhq2E)

[https://www.youtube.com/watch?v=VZXdkOo3xNI](https://www.youtube.com/watch?v=VZXdkOo3xNI)

[Our Chatgpt chat used for this](https://chatgpt.com/c/679a3621-0ecc-800b-ac2e-3420f235bbbc)