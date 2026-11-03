# Computer Vision: Vehicle Detection

เว็บแอปสำหรับตรวจจับประเภทรถจากภาพและวิดีโอ พัฒนาด้วย Python, OpenCV และ Streamlit โดยเรียกใช้โมเดลที่โฮสต์บน Roboflow ผ่าน Hosted API

## ความสามารถ

- อัปโหลดภาพ JPG, JPEG, PNG หรือ WEBP แล้วแสดงกรอบตรวจจับ
- อัปโหลดวิดีโอ MP4 หรือ MOV แล้วสร้างวิดีโอผลลัพธ์ MP4
- แสดงกรอบตรวจจับพร้อมประเภทที่จัดกลุ่มเป็น Car, Bus และ Truck รวมถึงค่า confidence แบบทศนิยม เช่น 0.86 พร้อมจำนวนที่ตรวจพบ
- ดาวน์โหลดภาพ/วิดีโอผลลัพธ์และไฟล์ CSV ที่มีประเภท พิกัด และ confidence จากหน้าแอป
- บันทึกไฟล์ต้นฉบับ ผลลัพธ์ และสถิติไว้ในโฟลเดอร์ของโปรเจกต์

## Workflow

```text
อัปโหลดภาพหรือวิดีโอ
        ↓
Streamlit ตรวจสอบไฟล์และเตรียมภาพ (ปรับขนาด / Resize)
        ↓
ส่งภาพหรือเฟรมตัวอย่างไปยัง Roboflow Hosted API
        ↓
โมเดลตรวจจับรถและส่งกรอบ/ประเภทวัตถุกลับมา
        ↓
แสดงกรอบและสถิติบนหน้าเว็บ
        ↓
บันทึกผลภาพ/วิดีโอและ CSV
```

### Image Processing

แอปปรับขนาดภาพก่อน inference โดยภาพมีด้านยาวไม่เกิน 1,280 พิกเซล และเฟรมวิดีโอที่ส่งให้โมเดลปรับขนาดไม่เกิน 1,280 × 1,280 โดยคงสัดส่วนภาพไว้ การวาดกรอบและชื่อคลาสเป็นการแสดงผลลัพธ์ของการตรวจจับ

## Model

- แหล่งโมเดล: [Roboflow Universe — Vehicles](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v)
- Model ID: `vehicles-q0x2v/1`
- ประเภท: YOLOv5 object detection (Roboflow Hosted API)
- โมเดลทำนาย 12 คลาสรถ และแอปรวมชื่อที่แสดงเป็น Car / Bus / Truck
- Confidence threshold ในแอปตั้งไว้ที่ `0.25`

การใช้งานต้องเชื่อมต่ออินเทอร์เน็ตและมี Roboflow API key ที่เข้าถึงโมเดลได้

## Dataset และ Evaluation

Dataset อ้างอิงจาก [Roboflow Universe — Dataset version 1](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/1) จำนวน 4,058 ภาพ แบ่งเป็น Train 2,634, Validation 966 และ Test 458 มี 12 คลาส และหน้า Dataset ระบุ license เป็น CC BY 4.0

Dataset ไม่ได้รวมอยู่ใน repository; แอปเรียกใช้ Hosted Model จึงไม่ต้องดาวน์โหลด Dataset เพื่อทดลอง inference แต่หากต้องส่ง Dataset ตามข้อกำหนดรายวิชา ให้ดาวน์โหลดชุดข้อมูลและแนบแหล่งที่มา/license ตามเงื่อนไข Roboflow

หน้า Roboflow รายงานค่า mAP@50 45.6%, Precision 55.5% และ Recall 53.9% ตัวเลขนี้เป็นค่าจากผู้เผยแพร่โมเดล ไม่ใช่ผลประเมินที่โปรเจกต์นี้รันซ้ำเอง ส่วน CSV ที่แอปสร้างเป็นผล inference ไม่ใช่การประเมินเทียบ ground-truth บน Test split

## การติดตั้งและรันบน Windows

ต้องติดตั้ง Python ก่อน จากนั้นเปิด PowerShell แล้วรัน:

```powershell
cd C:\Users\minee\Documents\GitHub\computer-vision-vehicle-detection
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

เปิดไฟล์ `.env` แล้วใส่ API key ของตัวเอง:

```dotenv
ROBOFLOW_API_KEY=ใส่_API_KEY_ของคุณ
ROBOFLOW_MODEL_ID=vehicles-q0x2v/1
```

เริ่มแอปด้วย:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

เปิด URL ที่ Streamlit แสดงใน Terminal โดยปกติคือ `http://localhost:8501` ห้าม commit หรือแชร์ `.env` และ API key

## ข้อจำกัดปัจจุบัน

- ภาพและวิดีโอ: ขนาดไฟล์ไม่เกิน 100 MB ต่อไฟล์; รับภาพและวิดีโอความละเอียดสูง เช่น 4K ได้
- วิดีโอ: MP4/MOV ความยาวไม่เกิน 20 วินาที; ย่อเฟรมก่อนส่งให้โมเดล และวิดีโอผลลัพธ์ย่อไม่เกิน 1920×1080
- วิดีโอเรียกโมเดลกับเฟรมตัวอย่างทุกประมาณ 0.2 วินาที; ผลกรอบล่าสุดจะถูกแสดงต่อเนื่องระหว่างเฟรมตัวอย่าง
- การนับใน CSV วิดีโอเป็นจำนวนที่ตรวจพบในแต่ละเฟรมตัวอย่าง รถคันเดิมอาจถูกนับซ้ำเมื่อนำหลายเฟรมมารวมกัน
- ระบบต้องเข้าถึง Roboflow API และอาจอยู่ภายใต้ quota/ข้อจำกัดของบริการ

## โครงสร้างโปรเจกต์

```text
computer-vision-vehicle-detection/
├── app.py              # จุดเริ่มรัน Streamlit
├── views/
│   ├── __init__.py
│   └── detection_view.py # หน้า UI และการควบคุม Streamlit
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── detection.py
│   └── preprocessing.py
├── images/
│   ├── input/       # ไฟล์ต้นฉบับจากการใช้งาน
│   └── results/     # ภาพ/วิดีโอที่มีกรอบตรวจจับ
├── results/         # CSV ผลตรวจจับและสถิติ
└── demo/            # พื้นที่สำหรับไฟล์วิดีโอสาธิต
```

## แหล่งอ้างอิง

- [Roboflow Universe — Vehicles model](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v)
- [Roboflow Dataset version 1](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/1)
- [Roboflow Inference SDK](https://inference.roboflow.com/quickstart/roboflow_ecosystem/)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

