import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
from pathlib import Path

# validasi input image x-ray thoraks 
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

# YOLOv11 inference
def predict_yolo(model, image: Image.Image, conf_threshold: float = 0.25):
    """
    jalankan inferensi YOLOv11 pada gambar PIL.
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

    # gambar dengan bounding box (BGR → RGB)
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

# SSD300 INFERENCE
CLASS_NAMES_SSD = ['background', 'kardiomegali', 'normal']
SSD_SIZE        = 300

_ssd_transform = T.Compose([
    T.Resize((SSD_SIZE, SSD_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

COLOR_MAP_SSD = {
    1: (255, 107, 107),   # kardiomegali — merah
    2: (78,  205, 196),   # normal       — teal
}


def predict_ssd(model, image: Image.Image,
                conf_threshold: float = 0.50,
                device: str = 'cpu'):
    """
    Jalankan inferensi SSD300 pada gambar PIL.
    Returns:
        result_image : PIL.Image dengan bounding box
        detections   : list of dict {'class_name', 'confidence', 'box': [x1,y1,x2,y2]}
    """
    device = torch.device(device)
    model.eval().to(device)

    orig = np.array(image.convert('RGB'))
    h, w = orig.shape[:2]
    draw = orig.copy()

    img_tensor = _ssd_transform(image.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    keep   = prediction['scores'] >= conf_threshold
    boxes  = prediction['boxes'][keep].cpu().numpy()
    scores = prediction['scores'][keep].cpu().numpy()
    labels = prediction['labels'][keep].cpu().numpy()

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        cls = int(label)
        # Skala dari 300×300 ke resolusi asli
        x1 = int(box[0] * w / SSD_SIZE)
        y1 = int(box[1] * h / SSD_SIZE)
        x2 = int(box[2] * w / SSD_SIZE)
        y2 = int(box[3] * h / SSD_SIZE)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cls_name = CLASS_NAMES_SSD[cls] if cls < len(CLASS_NAMES_SSD) else str(cls)
        color    = COLOR_MAP_SSD.get(cls, (200, 200, 200))

        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 3)

        text = f'{cls_name}: {score:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lbl_y1 = y1 - th - 10 if y1 - th - 10 >= 0 else y1
        lbl_y2 = lbl_y1 + th + 10
        cv2.rectangle(draw, (x1, lbl_y1), (x1 + tw + 10, lbl_y2), color, -1)
        cv2.putText(draw, text, (x1 + 5, lbl_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        detections.append({
            'class_name' : cls_name,
            'confidence' : float(score),
            'box'        : [x1, y1, x2, y2],
        })

    result_image = Image.fromarray(draw)
    return result_image, detections