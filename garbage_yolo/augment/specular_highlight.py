import cv2
import numpy as np
import albumentations as A

class SpecularHighlight(A.ImageOnlyTransform):
    def apply(self, image, **params):
        h, w, _ = image.shape
        overlay = image.copy()

        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(min(h, w)//12, min(h, w)//6)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)

        highlight = np.dstack([mask]*3)
        overlay = cv2.addWeighted(overlay, 1.0, highlight, 0.4, 0)

        return overlay

def specular_transform():
    return A.Compose([
        SpecularHighlight(p=0.35)
    ])
