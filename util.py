import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
from pathlib import Path


# VALIDASI X-RAY
def is_valid_xray(image: Image.Image) -> bool:
    try:
        gray = np.array(image.convert("L"))
        if gray.shape[0] < 200 or gray.shape[1] < 200:
            return False
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        dark   = hist[:40].sum()
        bright = hist[215:].sum()
        if bright > 0.35 or dark > 0.35:
            return False
        return True
    except:
        return False


# NORMALISASI OTOMATIS X-RAY (CLAHE)
def normalize_xray(image: Image.Image) -> Image.Image:
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    gray = cv2.resize(gray, (1024, 1024))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.resize(enhanced, (512, 512))
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)

# YOLOv11 INFERENCE
# Notebook menggunakan: YOLO('yolo11s.pt') dari ultralytics
# Output project: outputs/yolo11/thorax/weights/best.pt
def predict_yolo(model, image: Image.Image, conf_threshold: float = 0.25):
    """
    Jalankan inferensi YOLOv11 pada gambar PIL.
    YOLOv11 menggunakan API Ultralytics yang sama dengan YOLOv8,
    namun model dimuat dari checkpoint yolo11*.pt.

    Returns:
        result_image : PIL.Image dengan bounding box
        detections   : list of dict {'class_name', 'confidence', 'box': [x1,y1,x2,y2]}
    """
    results = model.predict(
        source=np.array(image),
        conf=conf_threshold,
        save=False,
        verbose=False,
    )

    res = results[0]

    # Gambar dengan bounding box (BGR → RGB)
    plotted = res.plot()
    result_image = Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))

    detections = []
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                'class_name' : res.names[cls_id],
                'confidence' : conf,
                'box'        : xyxy,
            })

    return result_image, detections


# ============================================================
# SSD300 INFERENCE
# Notebook: NUM_CLASSES=3 → ['background','kardiomegali','normal']
# Checkpoint: outputs/ssd_best.pt
# Label index dari SSD: 1=kardiomegali, 2=normal
# ============================================================

# Sesuai notebook: CLASS_NAMES = ['kardiomegali', 'normal']
# SSD labels: 0=background, 1=kardiomegali, 2=normal
CLASS_NAMES_SSD = ['background', 'kardiomegali', 'normal']
SSD_SIZE        = 300   # sesuai notebook SSD_SIZE = 300

_ssd_transform = T.Compose([
    T.Resize((SSD_SIZE, SSD_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Sesuai notebook colors_map: {0: merah (kardiomegali), 1: teal (normal)}
# Namun label SSD dimulai dari 1, jadi kita map: cls-1 → warna
COLOR_MAP_SSD = {
    1: (255, 107, 107),   # kardiomegali — merah
    2: (78,  205, 196),   # normal       — teal
}


def predict_ssd(model, image: Image.Image,
                conf_threshold: float = 0.50,
                device: str = 'cpu'):
    """
    Jalankan inferensi SSD300 pada gambar PIL.
    Sesuai dengan notebook: label 1=kardiomegali, 2=normal.
    Bounding box SSD di-scale ke ukuran gambar asli.

    Returns:
        result_image : PIL.Image dengan bounding box
        detections   : list of dict {'class_name', 'confidence', 'box': [x1,y1,x2,y2]}
    """
    device_obj = torch.device(device)
    model.eval().to(device_obj)

    orig = np.array(image.convert('RGB'))
    h, w = orig.shape[:2]
    draw = orig.copy()

    img_tensor = _ssd_transform(image.convert('RGB')).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    keep   = prediction['scores'] >= conf_threshold
    boxes  = prediction['boxes'][keep].cpu().numpy()
    scores = prediction['scores'][keep].cpu().numpy()
    labels = prediction['labels'][keep].cpu().numpy()

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        cls = int(label)

        # Scale bounding box dari SSD_SIZE (300) ke ukuran gambar asli
        # Sesuai logika di notebook: x1 = int(x1 * w / SSD_SIZE)
        x1 = int(box[0] * w / SSD_SIZE)
        y1 = int(box[1] * h / SSD_SIZE)
        x2 = int(box[2] * w / SSD_SIZE)
        y2 = int(box[3] * h / SSD_SIZE)

        # Clamp supaya tidak keluar gambar
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cls_name = CLASS_NAMES_SSD[cls] if cls < len(CLASS_NAMES_SSD) else str(cls)
        color    = COLOR_MAP_SSD.get(cls, (200, 200, 200))

        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 3)

        text = f'{cls_name}: {score:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Sesuai notebook: fix label position jika terlalu dekat atas
        if y1 - th - 10 < 0:
            lbl_y1 = y1
            lbl_y2 = y1 + th + 10
            text_y = y1 + th + 5
        else:
            lbl_y1 = y1 - th - 10
            lbl_y2 = y1
            text_y = y1 - 5

        cv2.rectangle(draw, (x1, lbl_y1), (x1 + tw + 10, lbl_y2), color, -1)
        cv2.putText(draw, text, (x1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        detections.append({
            'class_name' : cls_name,
            'confidence' : float(score),
            'box'        : [x1, y1, x2, y2],
        })

    result_image = Image.fromarray(draw)
    return result_image, detections