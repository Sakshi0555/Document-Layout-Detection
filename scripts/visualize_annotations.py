import json
import os
import cv2

# -----------------------------
# Paths
# -----------------------------
TRAIN_FOLDER = "dataset/train"
ANNOTATION_FILE = os.path.join(TRAIN_FOLDER, "_annotations.coco.json")

# -----------------------------
# Load COCO JSON
# -----------------------------
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

# -----------------------------
# Categories Dictionary
# -----------------------------
categories = {}
for cat in coco["categories"]:
    categories[cat["id"]] = cat["name"]

# -----------------------------
# First Image
# -----------------------------
image_number = 500      # Change this number whenever you want

image_info = coco["images"][image_number]

image_name = image_info["file_name"]
image_id = image_info["id"]

image_path = os.path.join(TRAIN_FOLDER, image_name)

image = cv2.imread(image_path)

# -----------------------------
# Draw Bounding Boxes
# -----------------------------
for ann in coco["annotations"]:

    if ann["image_id"] != image_id:
        continue

    x, y, w, h = ann["bbox"]

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    class_name = categories[ann["category_id"]]

    colors = {
    "text": (0,255,0),
    "caption": (255,0,255),
    "chapter-heading": (0,0,255),
    "header": (255,255,0),
    "image": (255,0,0),
    "list": (0,255,255),
    "section-heading": (0,165,255),
    "sub-section-heading": (128,0,255),
    "table": (255,255,255)
    }

    color = colors.get(class_name, (0,255,0))

    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    cv2.putText(
    image,
    class_name,
    (x, y-5),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    color,
    2
)

# -----------------------------
# Show Image
# -----------------------------
cv2.imshow("Document Layout", image)

cv2.waitKey(0)

cv2.destroyAllWindows()