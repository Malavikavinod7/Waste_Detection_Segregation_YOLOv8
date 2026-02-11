import os
import cv2
from glob import glob
import albumentations as A

from color_space import color_space_transform
from texture_emphasis import texture_emphasis
from specular_highlight import specular_transform
from partial_visibility import partial_visibility

# ---------------- PATHS ----------------
IMG_IN = "data/raw/GARBAGE_CLASSIFICATION/train/images"
LBL_IN = "data/raw/GARBAGE_CLASSIFICATION/train/labels"

IMG_OUT = "data/augmented/train/images"
LBL_OUT = "data/augmented/train/labels"

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# ---------------- PIPELINE ----------------
transform = A.Compose(
    [
        color_space_transform(),
        texture_emphasis(),
        specular_transform(),
        partial_visibility(),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["labels"])
)

# ---------------- RUN ----------------
for img_path in glob(f"{IMG_IN}/*.jpg"):
    name = os.path.basename(img_path)
    label_path = os.path.join(LBL_IN, name.replace(".jpg", ".txt"))

    image = cv2.imread(img_path)

    boxes, labels = [], []
    with open(label_path) as f:
        for line in f:
            c, x, y, w, h = map(float, line.split())
            boxes.append([x, y, w, h])
            labels.append(int(c))

    augmented = transform(
        image=image,
        bboxes=boxes,
        labels=labels
    )

    cv2.imwrite(os.path.join(IMG_OUT, name), augmented["image"])

    with open(os.path.join(LBL_OUT, name.replace(".jpg", ".txt")), "w") as f:
        for c, b in zip(augmented["labels"], augmented["bboxes"]):
            f.write(f"{c} {' '.join(map(str, b))}\n")
