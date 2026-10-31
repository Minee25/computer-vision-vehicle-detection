from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_INFERENCE_SIDE = 1280


def vehicle_group(class_name: str) -> str:
    """Map detailed model classes to simple display groups."""
    name = str(class_name).strip().lower()
    if name == "car":
        return "Car"
    if "bus" in name:
        return "Bus"
    if "truck" in name:
        return "Truck"
    return "Other"


def load_image(uploaded_file) -> np.ndarray:
    """Read an uploaded image, apply EXIF orientation, and return RGB pixels."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((MAX_INFERENCE_SIDE, MAX_INFERENCE_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(image)


def resize_frame(frame_bgr: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
    """Keep aspect ratio while resizing video frames to a browser-friendly size."""
    height, width = frame_bgr.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return frame_bgr.copy()
    new_size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def save_rgb_temp(image_rgb: np.ndarray) -> str:
    """Write an RGB array to a temporary JPEG for the Roboflow SDK."""
    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    path = handle.name
    handle.close()
    Image.fromarray(image_rgb).save(path, format="JPEG", quality=92)
    return path


def draw_predictions(image_rgb: np.ndarray, predictions: list[dict], confidence_threshold: float = 0.0) -> np.ndarray:
    """Draw Roboflow center-based boxes on an RGB image and return RGB pixels."""
    output = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    height, width = output.shape[:2]
    thickness = max(2, round(max(width, height) / 500))

    for prediction in predictions:
        confidence = float(prediction.get("confidence", 0) or 0)
        if confidence < confidence_threshold:
            continue
        try:
            cx = float(prediction["x"])
            cy = float(prediction["y"])
            box_w = float(prediction["width"])
            box_h = float(prediction["height"])
        except (KeyError, TypeError, ValueError):
            continue

        x1 = max(0, min(width - 1, round(cx - box_w / 2)))
        y1 = max(0, min(height - 1, round(cy - box_h / 2)))
        x2 = max(0, min(width - 1, round(cx + box_w / 2)))
        y2 = max(0, min(height - 1, round(cy + box_h / 2)))
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = vehicle_group(str(prediction.get("class", "vehicle")))
        color = (34, 197, 94)
        label = class_name
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        text_y = max(20, y1 - 7)
        cv2.putText(output, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def predictions_to_rows(predictions: list[dict], source: str, second: float | None = None) -> list[dict]:
    rows = []
    for prediction in predictions:
        rows.append({
            "source": source,
            "second": None if second is None else round(second, 2),
            "vehicle_type": vehicle_group(str(prediction.get("class", "unknown"))),
            "x_center": prediction.get("x"),
            "y_center": prediction.get("y"),
            "width": prediction.get("width"),
            "height": prediction.get("height"),
        })
    return rows
