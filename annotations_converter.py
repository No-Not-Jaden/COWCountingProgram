
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil

# ---------------- CONFIGURATION ----------------

XML_FILE = Path("data/kagglehub/annotations.xml")
IMAGE_ROOT = Path("data/kagglehub")

OUTPUT_DIR = Path("dataset/cow_dataset")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Include cows marked is_visible=false?
INCLUDE_NOT_VISIBLE = True

# ------------------------------------------------

random.seed(RANDOM_SEED)

# Parse XML
tree = ET.parse(XML_FILE)
root = tree.getroot()

# Collect images and annotations
images = []

for image in root.findall("image"):

    image_name = image.get("name")
    width = float(image.get("width"))
    height = float(image.get("height"))

    boxes = []

    for box in image.findall("box"):

        if box.get("label") != "cow":
            continue

        # Check visibility attribute
        visible_attr = box.find("attribute[@name='is_visible']")

        if visible_attr is not None:
            is_visible = visible_attr.text.lower() == "true"

            if not is_visible and not INCLUDE_NOT_VISIBLE:
                continue

        # Read bounding box
        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))

        # Normalize coordinates
        x_center = ((xtl + xbr) / 2) / width
        y_center = ((ytl + ybr) / 2) / height

        box_width = (xbr - xtl) / width
        box_height = (ybr - ytl) / height

        boxes.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    images.append({
        "name": image_name,
        "boxes": boxes
    })

# Shuffle and split dataset
random.shuffle(images)

total = len(images)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

# Create directories and copy files
for split_name, split_images in splits.items():

    image_dir = OUTPUT_DIR / "images" / split_name
    label_dir = OUTPUT_DIR / "labels" / split_name

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for item in split_images:

        source = IMAGE_ROOT / item["name"]

        if not source.exists():
            print(f"Missing image: {source}")
            continue

        # Copy image
        destination = image_dir / Path(item["name"]).name
        shutil.copy2(source, destination)

        # Write YOLO annotations
        label_file = label_dir / (destination.stem + ".txt")

        label_file.write_text(
            "\n".join(item["boxes"]),
            encoding="utf-8"
        )

# Generate YAML configuration
yaml_content = f"""path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: cow
"""

(OUTPUT_DIR / "cows.yaml").write_text(
    yaml_content,
    encoding="utf-8"
)

print("Conversion complete!")
print(f"Total images: {total}")

for split_name, split_images in splits.items():
    print(f"{split_name}: {len(split_images)} images")

print(f"Output directory: {OUTPUT_DIR}")
