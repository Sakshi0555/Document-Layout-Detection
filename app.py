import easyocr
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load trained model
model = YOLO("runs/detect/runs/document_layout-2/weights/best.pt")
reader = easyocr.Reader(['en'])

st.set_page_config(page_title="Document Layout Detection")

st.title("📄 Intelligent Document Layout Detection")

uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        # Predict
        results = model.predict(tmp.name, conf=0.4)

    # Plot prediction
    result_image = results[0].plot()

    st.subheader("Detected Layout")

    st.image(result_image, use_container_width=True)
    st.subheader("Extracted Text")

boxes = results[0].boxes

image_np = np.array(image)

for box in boxes:

    cls = int(box.cls[0])

    class_name = model.names[cls]

    # Only perform OCR on text-related regions
    if class_name in [
        "text",
        "caption",
        "chapter-heading",
        "header",
        "section-heading",
        "sub-section-heading"
    ]:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = image_np[y1:y2, x1:x2]

        text = reader.readtext(crop, detail=0)

        st.write(f"### {class_name}")

        if len(text):
            st.write(" ".join(text))
        else:
            st.write("No text found.")