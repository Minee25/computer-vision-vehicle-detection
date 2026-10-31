from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

from .preprocessing import draw_predictions, predictions_to_rows, resize_frame, save_rgb_temp, vehicle_group

MODEL_ID_DEFAULT = "vehicles-q0x2v/1"
ALLOWED_CLASSES = {
    "car", "big bus", "big truck", "bus-l-", "bus-s-", "mid truck",
    "small bus", "small truck", "truck-l-", "truck-m-", "truck-s-", "truck-xl-",
}


def create_client(api_key: str) -> InferenceHTTPClient:
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
    return client.configure(InferenceConfiguration(api_key_transport="header"))


def infer_rgb(client: InferenceHTTPClient, image_rgb: np.ndarray, model_id: str = MODEL_ID_DEFAULT) -> dict:
    """Call the hosted Roboflow model with an RGB image."""
    path = save_rgb_temp(image_rgb)
    try:
        return client.infer(path, model_id=model_id)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def infer_image(client: InferenceHTTPClient, image_rgb: np.ndarray, model_id: str, confidence_threshold: float) -> tuple[np.ndarray, list[dict]]:
    result = infer_rgb(client, image_rgb, model_id)
    predictions = [
        item for item in result.get("predictions", [])
        if float(item.get("confidence", 0) or 0) >= confidence_threshold
    ]
    return draw_predictions(image_rgb, predictions), predictions


def process_video(
    client: InferenceHTTPClient,
    input_path: str,
    output_path: str,
    model_id: str,
    confidence_threshold: float = 0.25,
    sample_seconds: float = 1.0,
    max_seconds: float = 10.0,
) -> tuple[list[dict], list[dict]]:
    """Sample a video, annotate every frame with the latest inference, and encode H.264."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not read this video. Try an MP4 encoded with H.264.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0 or fps > 60:
        cap.release()
        raise ValueError("Unsupported video properties. Video must have valid FPS and be at most 60 FPS.")
    duration = frame_count / fps
    if duration > max_seconds:
        cap.release()
        raise ValueError(f"Video is {duration:.1f}s; the demo limit is {max_seconds:g}s.")

    output_scale = min(1.0, 1920 / width, 1080 / height)
    out_w = max(2, int(width * output_scale) // 2 * 2)
    out_h = max(2, int(height * output_scale) // 2 * 2)
    raw_path = str(Path(output_path).with_name(Path(output_path).stem + "_raw.mp4"))
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the temporary output video.")

    interval = max(1, round(fps * sample_seconds))
    latest_predictions: list[dict] = []
    latest_inference_width = 1
    latest_inference_height = 1
    sample_rows: list[dict] = []
    prediction_rows: list[dict] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_out = resize_frame(frame)

            if frame_index % interval == 0:
                inference_frame = resize_frame(frame, 1280, 1280)
                rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                latest_inference_height, latest_inference_width = rgb.shape[:2]
                result = infer_rgb(client, rgb, model_id)
                latest_predictions = [
                    item for item in result.get("predictions", [])
                    if float(item.get("confidence", 0) or 0) >= confidence_threshold
                ]
                second = frame_index / fps
                counts: dict[str, int] = {}
                for item in latest_predictions:
                    class_name = vehicle_group(str(item.get("class", "unknown")))
                    counts[class_name] = counts.get(class_name, 0) + 1
                sample_rows.append({"second": round(second, 2), "total_detections": len(latest_predictions), **counts})
                prediction_rows.extend(predictions_to_rows(latest_predictions, Path(input_path).name, second))

            # Inference coordinates use the resized frame. Scale boxes for the displayed output frame.
            # Scale boxes from the last sampled inference frame to the output video frame.
            scale_x = out_w / latest_inference_width
            scale_y = out_h / latest_inference_height
            scaled_predictions = []
            for item in latest_predictions:
                scaled_predictions.append({
                    **item,
                    "x": float(item.get("x", 0)) * scale_x,
                    "y": float(item.get("y", 0)) * scale_y,
                    "width": float(item.get("width", 0)) * scale_x,
                    "height": float(item.get("height", 0)) * scale_y,
                })
            frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            annotated_rgb = draw_predictions(frame_rgb, scaled_predictions)
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            frame_index += 1
    except Exception:
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise
    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        raise ValueError("No readable frames were found in this video.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    command = [
        ffmpeg, "-y", "-i", raw_path, "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", output_path,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg failed to create browser-compatible MP4: {completed.stderr[-1000:]}")
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass

    return sample_rows, prediction_rows



