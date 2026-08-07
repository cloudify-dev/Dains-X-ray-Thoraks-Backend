# uvicorn api:app --reload --host 0.0.0.0 --port 8001

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, base64, torch
from ultralytics import YOLO
from utils import is_valid_xray, normalize_xray, predict_yolo, predict_ssd
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

# load model
YOLO_PATH = "runs/detect/outputs/yolo11/thorax/weights/best.pt"
SSD_PATH  = "outputs/ssd_best.pt"

yolo_model = YOLO(YOLO_PATH)

ssd_model = ssd300_vgg16(weights=None, score_thresh=0.01, nms_thresh=0.45, detections_per_img=10)
in_channels = [512, 1024, 512, 256, 256, 256]
num_anchors = ssd_model.anchor_generator.num_anchors_per_location()
ssd_model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, 3)
ssd_model.load_state_dict(torch.load(SSD_PATH, map_location="cpu", weights_only=True))
ssd_model.eval()

# FastAPI App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    model: str = Form("yolo"),
    conf: float = Form(0.25),
):
    # 1. Baca gambar
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Validasi apakah ini X-ray
    if not is_valid_xray(image):
        return {"error": "Gambar bukan chest X-ray yang valid"}

    # 3. CLAHE enhancement
    processed = normalize_xray(image)

    # 4. Jalankan model
    if model == "yolo":
        result_img, detections = predict_yolo(yolo_model, processed, conf)
    else:
        result_img, detections = predict_ssd(ssd_model, processed, conf, "cpu")

    # 5. Ubah gambar hasil jadi base64
    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "result_image_b64": img_b64,
        "detections": detections,
    }

@app.get("/health")
def health():
    return {"status": "ok"}