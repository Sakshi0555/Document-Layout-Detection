from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/runs/document_layout-2/weights/best.pt")

# Image to test
image_path = "p1p.jpeg"

# Run prediction
results = model.predict(
    source=image_path,
    conf=0.4,
    save=True
)

print("Prediction completed!")