import json
import os
from collections import defaultdict

DATASET_FOLDER = "dataset"

for split in ["train", "valid", "test"]:

    print(f"\nProcessing {split}...")

    SPLIT_FOLDER = os.path.join(DATASET_FOLDER, split)
    ANNOTATION_FILE = os.path.join(SPLIT_FOLDER, "_annotations.coco.json")
    LABELS_FOLDER = os.path.join(SPLIT_FOLDER, "labels")

    os.makedirs(LABELS_FOLDER, exist_ok=True)

    with open(ANNOTATION_FILE, "r") as f:
        coco = json.load(f)

    # Image dictionary
    images = {img["id"]: img for img in coco["images"]}

    # Store labels grouped by image
    labels = defaultdict(list)

    # Convert annotations
    for ann in coco["annotations"]:

        image = images[ann["image_id"]]

        img_w = image["width"]
        img_h = image["height"]

        x, y, w, h = ann["bbox"]

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h

        w /= img_w
        h /= img_h

        image_name = os.path.splitext(image["file_name"])[0]

        # Category ID
        class_id = ann["category_id"]

        # Merge duplicate text class
        if class_id == 9:
            class_id = 0

        labels[image_name].append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

    # Write one txt file per image
    for image_name, lines in labels.items():

        label_path = os.path.join(LABELS_FOLDER, image_name + ".txt")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"{split} completed!")

print("\nAll datasets converted successfully!")