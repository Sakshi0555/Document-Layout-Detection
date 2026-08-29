import os

LABELS_FOLDER = "dataset/train/labels"

files = os.listdir(LABELS_FOLDER)

print("Total Label Files:", len(files))

print("\nFirst 5 Label Files:")

for file in files[:5]:
    print(file)

print("\nContents of First Label File:\n")

with open(os.path.join(LABELS_FOLDER, files[0]), "r") as f:
    print(f.read())