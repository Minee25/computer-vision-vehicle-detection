from __future__ import annotations

import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from PIL import Image

from src.detection import MODEL_ID_DEFAULT, create_client, infer_image, process_video
from src.preprocessing import load_image, vehicle_group

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", MODEL_ID_DEFAULT)
MAX_IMAGE_MB = 10
MAX_VIDEO_MB = 30
MAX_VIDEO_SECONDS = 10
INPUT_DIR = ROOT / "images" / "input"
IMAGE_RESULTS_DIR = ROOT / "images" / "results"
RESULTS_DIR = ROOT / "results"
for output_dir in (INPUT_DIR, IMAGE_RESULTS_DIR, RESULTS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    st.set_page_config(page_title="Vehicles Detection", page_icon="🚗", layout="wide")


    @st.cache_resource(show_spinner=False)
    def cached_client(api_key: str) -> InferenceHTTPClient:
        return create_client(api_key)


    def get_saved_key() -> str:
        try:
            return str(st.secrets.get("ROBOFLOW_API_KEY", ""))
        except Exception:
            return os.getenv("ROBOFLOW_API_KEY", "")


    def count_by_class(predictions: list[dict]) -> pd.DataFrame:
        counts = Counter(vehicle_group(str(item.get("class", "unknown"))) for item in predictions)
        return pd.DataFrame([{"Vehicle type": name, "Count": count} for name, count in sorted(counts.items())])


    def make_run_id(filename: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._")[:60] or "upload"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stem}_{stamp}_{uuid4().hex[:6]}"


    def save_output(directory: Path, filename: str, content: bytes) -> str:
        path = directory / filename
        path.write_bytes(content)
        return path.relative_to(ROOT).as_posix()


    def inspect_video(path: str) -> tuple[float, float, int, int, int]:
        cap = cv2.VideoCapture(path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps if fps > 0 else 0
            return duration, fps, frames, width, height
        finally:
            cap.release()


    st.title("🚗 Vehicles Detection")
    st.caption("ตรวจจับและนับประเภทรถจากภาพหรือวิดีโอด้วย Roboflow Hosted Model")

    with st.sidebar:
        st.header("ตั้งค่าการตรวจจับ")
        api_key = st.text_input("Roboflow API Key", value=get_saved_key(), type="password", help="ใส่ API key จากบัญชี Roboflow ของคุณ")
        st.text_input("Model ID", value=MODEL_ID, disabled=True)
        confidence = 0.25
        st.caption("Detection threshold: 0.25")
        st.divider()
        st.caption("Image: สูงสุด 10 MB")
        st.caption("Video: MP4/MOV, สูงสุด 30 MB และ 10 วินาที")
        st.caption("วิดีโอจะสุ่มตรวจเฟรมทุกประมาณ 0.2 วินาที")

    if not api_key:
        st.info("ใส่ Roboflow API Key ในแถบด้านซ้ายเพื่อเริ่มตรวจจับ")

    uploaded = st.file_uploader(
        "เลือกรูปภาพหรือวิดีโอ",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov"],
        help="รองรับ JPG/PNG/WEBP และ MP4/MOV",
    )

    if uploaded:
        file_size_mb = uploaded.size / (1024 * 1024)
        extension = Path(uploaded.name).suffix.lower()
        is_video = extension in {".mp4", ".mov"}
        max_size_mb = MAX_VIDEO_MB if is_video else MAX_IMAGE_MB
        if file_size_mb > max_size_mb:
            st.error(f"ไฟล์มีขนาด {file_size_mb:.2f} MB เกินขีดจำกัด {max_size_mb} MB")
            st.stop()

        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            st.subheader("ไฟล์ต้นฉบับ")
            if not is_video:
                st.image(uploaded, width="stretch")
            else:
                video_bytes = uploaded.getvalue()
                st.video(video_bytes)
                temp_input = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
                temp_input.write(video_bytes)
                temp_input.close()
                try:
                    duration, fps, frames, width, height = inspect_video(temp_input.name)
                finally:
                    try:
                        os.remove(temp_input.name)
                    except OSError:
                        pass
                st.caption(f"{width}×{height} · {fps:.1f} FPS · {duration:.2f} วินาที · {file_size_mb:.2f} MB")
                if duration <= 0 or fps > 60:
                    st.error("อ่านข้อมูลวิดีโอไม่ได้ หรือ FPS สูงกว่า 60")
                elif duration > MAX_VIDEO_SECONDS:
                    st.warning(f"วิดีโอยาวเกิน {MAX_VIDEO_SECONDS} วินาที กรุณาตัดคลิปให้สั้นลงก่อน")

        with right:
            st.subheader("ผลตรวจจับ")
            run_disabled = not api_key or (is_video and (duration <= 0 or duration > MAX_VIDEO_SECONDS or fps > 60))
            if st.button("เริ่มตรวจจับ", type="primary", disabled=run_disabled, width="stretch"):
                client = cached_client(api_key)
                run_id = make_run_id(uploaded.name)
                try:
                    if not is_video:
                        rgb = load_image(uploaded)
                        with st.spinner("กำลังส่งภาพไปยัง Roboflow..."):
                            annotated, predictions = infer_image(client, rgb, MODEL_ID, confidence)
                        class_counts = count_by_class(predictions)
                        st.image(annotated, caption="ภาพพร้อมกรอบตรวจจับ", width="stretch")
                        st.metric("จำนวนรถที่ตรวจพบ", len(predictions))
                        if not class_counts.empty:
                            st.dataframe(class_counts, hide_index=True, width="stretch")
                        else:
                            st.info("ไม่พบวัตถุที่ผ่านเกณฑ์ตรวจจับ")
                        detection_rows = pd.DataFrame([
                            {
                                "vehicle_type": vehicle_group(str(item.get("class", "unknown"))),
                                "x_center": item.get("x"), "y_center": item.get("y"),
                                "width": item.get("width"), "height": item.get("height"),
                            }
                            for item in predictions
                        ])
                        st.download_button("ดาวน์โหลดผลเป็น CSV", detection_rows.to_csv(index=False).encode("utf-8-sig"), "detections.csv", "text/csv")
                        png = Image.fromarray(annotated)
                        import io
                        buffer = io.BytesIO()
                        png.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        input_saved = save_output(INPUT_DIR, f"{run_id}{extension}", uploaded.getvalue())
                        image_saved = save_output(IMAGE_RESULTS_DIR, f"{run_id}_detected.png", image_bytes)
                        csv_saved = save_output(RESULTS_DIR, f"{run_id}_detections.csv", detection_rows.to_csv(index=False).encode("utf-8-sig"))
                        st.success(f"บันทึกไฟล์แล้ว: {input_saved} · {image_saved} · {csv_saved}")
                        st.download_button("ดาวน์โหลดภาพผลตรวจจับ", image_bytes, "detection_result.png", "image/png")
                    else:
                        input_temp = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
                        input_temp.write(uploaded.getvalue())
                        input_temp.close()
                        output_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                        output_path = output_temp.name
                        output_temp.close()
                        try:
                            progress = st.progress(0, text="กำลังประมวลผลเฟรมและเรียก Roboflow...")
                            sample_rows, detection_rows = process_video(
                                client,
                                input_temp.name,
                                output_path,
                                MODEL_ID,
                                confidence_threshold=confidence,
                                sample_seconds=0.2,
                                max_seconds=MAX_VIDEO_SECONDS,
                            )
                            progress.progress(100, text="ประมวลผลวิดีโอเสร็จแล้ว")
                            output_bytes = Path(output_path).read_bytes()
                        finally:
                            for path in (input_temp.name, output_path):
                                try:
                                    os.remove(path)
                                except OSError:
                                    pass
                        input_saved = save_output(INPUT_DIR, f"{run_id}{extension}", uploaded.getvalue())
                        video_saved = save_output(IMAGE_RESULTS_DIR, f"{run_id}_detected.mp4", output_bytes)
                        st.video(output_bytes)
                        st.download_button("ดาวน์โหลดวิดีโอผลลัพธ์ MP4", output_bytes, "vehicles_detected.mp4", "video/mp4")
                        samples = pd.DataFrame(sample_rows).fillna(0)
                        if not samples.empty:
                            st.metric("เฉลี่ยต่อเฟรมตัวอย่าง", f"{samples['total_detections'].mean():.1f} detections")
                            st.metric("สูงสุดในเฟรมตัวอย่าง", int(samples["total_detections"].max()))
                            st.caption("จำนวนรวมในตารางนับซ้ำได้ระหว่างเฟรมตัวอย่าง จึงไม่ใช่จำนวนรถที่ไม่ซ้ำกันทั้งคลิป")
                            st.dataframe(samples, hide_index=True, width="stretch")
                            frame_csv = samples.to_csv(index=False).encode("utf-8-sig")
                            frame_saved = save_output(RESULTS_DIR, f"{run_id}_frame_statistics.csv", frame_csv)
                            st.download_button("ดาวน์โหลดสถิติแต่ละเฟรม CSV", frame_csv, "video_frame_statistics.csv", "text/csv")
                        detections = pd.DataFrame(detection_rows, columns=["source", "second", "vehicle_type", "x_center", "y_center", "width", "height"])
                        detection_csv = detections.to_csv(index=False).encode("utf-8-sig")
                        detection_saved = save_output(RESULTS_DIR, f"{run_id}_detections.csv", detection_csv)
                        st.success(f"บันทึกไฟล์แล้ว: {input_saved} · {video_saved} · {frame_saved if not samples.empty else ''} · {detection_saved}")
                        if not detections.empty:
                            st.download_button("ดาวน์โหลด Bounding Boxes CSV", detection_csv, "video_detections.csv", "text/csv")
                except Exception as error:
                    st.error(f"ตรวจจับไม่สำเร็จ: {error}")
                    st.caption("ตรวจ API key, Internet connection, Roboflow quota และ file format")




