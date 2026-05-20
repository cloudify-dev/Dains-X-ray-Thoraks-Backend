"""
app.py — CardioDetect
Deteksi Kardiomegali X-Ray Thoraks
YOLOv11s + SSD300 VGG16
"""

import time
import torch
import streamlit as st
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from streamlit_option_menu import option_menu

from util import (
    is_valid_xray,
    normalize_xray,
    predict_yolo,
    predict_ssd,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CardioDetect",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Font */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Main Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
}

/* Card */
.custom-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #30363D;
    margin-bottom: 20px;
}

/* Result Box */
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    border: 1px solid #30363D;
}

.result-normal {
    background-color: rgba(46, 160, 67, 0.15);
    border-left: 4px solid #2EA043 !important;
}

.result-cardio {
    background-color: rgba(248, 81, 73, 0.15);
    border-left: 4px solid #F85149 !important;
}

/* Info table inside result box */
.result-box table {
    width: 100%;
    margin-top: 10px;
    font-size: 0.9rem;
    border-collapse: collapse;
}

.result-box table tr td {
    padding: 5px 4px;
    vertical-align: top;
}

.result-box table tr td:first-child {
    color: #8B949E;
    width: 42%;
}

/* Stats table inside custom-card */
.custom-card table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

.custom-card table tr td {
    padding: 6px 4px;
    border-bottom: 1px solid #21262D;
}

.custom-card table tr:last-child td {
    border-bottom: none;
}

.custom-card table tr td:first-child {
    color: #8B949E;
    width: 55%;
}

/* Metric */
.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}

/* Upload */
[data-testid="stFileUploader"] {
    background-color: #161B22;
    border: 2px dashed #30363D;
    border-radius: 10px;
    padding: 10px;
}

/* Button */
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 10px;
    border: none;
}

/* Progress */
.stProgress > div > div > div > div {
    background-color: #238636;
}

/* Section header */
.section-header {
    background: linear-gradient(135deg, #161B22, #1C2128);
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 16px;
}

/* Divider label */
.divider-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #8B949E;
    font-size: 0.85rem;
    margin: 8px 0;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_yolo_model(path: str):
    return YOLO(path)


@st.cache_resource
def load_ssd_model(path: str, device: str):

    from torchvision.models.detection import ssd300_vgg16
    from torchvision.models.detection.ssd import SSDClassificationHead

    NUM_CLASSES = 3

    model = ssd300_vgg16(
        weights=None,
        score_thresh=0.01,
        nms_thresh=0.45,
        detections_per_img=10
    )

    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
    )

    model.load_state_dict(
        torch.load(
            path,
            map_location=torch.device(device),
            weights_only=True
        )
    )

    model.eval()

    return model


# ============================================================
# DEVICE
# ============================================================
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


# ============================================================
# MODEL PATH
# ============================================================
YOLO_WEIGHT = 'runs/detect/outputs/yolo11/thorax/weights/best.pt'
SSD_WEIGHT  = 'outputs/ssd_best.pt'

yolo_available = Path(YOLO_WEIGHT).exists()
ssd_available  = Path(SSD_WEIGHT).exists()

if yolo_available:
    yolo_model = load_yolo_model(YOLO_WEIGHT)

if ssd_available:
    ssd_model = load_ssd_model(SSD_WEIGHT, DEVICE)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown("""
    <div style='text-align:center'>
        <h1>🫀 CardioDetect</h1>
        <p style='color:gray'>
            Thorax X-ray Cardiomegaly Detection
        </p>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Home",
            "Detection",
            "Model Comparison",
            "About"
        ],
        icons=[
            "house",
            "activity",
            "bar-chart",
            "info-circle"
        ],
        menu_icon="heart-pulse",
        default_index=0,
    )

    st.markdown("---")

    st.markdown("### 📦 Model Status")

    if yolo_available:
        st.success("YOLOv11 Loaded ✅")
    else:
        st.error("YOLOv11 Not Found ❌")

    if ssd_available:
        st.success("SSD300 Loaded ✅")
    else:
        st.error("SSD300 Not Found ❌")

    st.markdown("---")

    st.info(f"🖥️ Device: {DEVICE.upper()}")


# ============================================================
# HELPER — render result card
# ============================================================
def render_result_card(det):
    """Tampilkan satu kartu hasil deteksi dengan interpretasi."""
    cls_name = det['class_name']
    conf     = det['confidence']
    pct      = conf * 100

    if cls_name.lower() == "kardiomegali":
        status_icon  = "🔴"
        status_label = "KARDIOMEGALI TERDETEKSI"
        box_class    = "result-cardio"
        interp       = (
            "Ukuran jantung terdeteksi <b>melebihi batas normal</b>. "
            "Segera konsultasikan dengan dokter spesialis jantung."
        )
        confidence_note = "Tinggi" if pct >= 70 else ("Sedang" if pct >= 40 else "Rendah")
    else:
        status_icon  = "🟢"
        status_label = "NORMAL"
        box_class    = "result-normal"
        interp       = (
            "Ukuran jantung dalam <b>batas normal</b>. "
            "Tidak ditemukan indikasi kardiomegali."
        )
        confidence_note = "Tinggi" if pct >= 70 else ("Sedang" if pct >= 40 else "Rendah")

    st.markdown(f"""
    <div class="result-box {box_class}">
        <h3>{status_icon} {status_label}</h3>
        <table>
            <tr>
                <td>Kelas Deteksi</td>
                <td><b>{cls_name}</b></td>
            </tr>
            <tr>
                <td>Confidence Score</td>
                <td><b>{pct:.2f}%</b> &nbsp;<span style="color:#8B949E;font-size:0.8rem;">({confidence_note})</span></td>
            </tr>
            <tr>
                <td>Interpretasi</td>
                <td>{interp}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(conf))


def render_stats_card(model_name: str, det_count: int, inf_time: float, threshold: float):
    """Tampilkan kartu statistik inferensi."""
    speed_note = "Cepat ⚡" if inf_time < 500 else ("Normal 🕐" if inf_time < 1500 else "Lambat 🐢")
    st.markdown(f"""
    <div class="custom-card">
        <table>
            <tr>
                <td>Model</td>
                <td><b>{model_name}</b></td>
            </tr>
            <tr>
                <td>Objek Terdeteksi</td>
                <td><b>{det_count}</b></td>
            </tr>
            <tr>
                <td>Waktu Inferensi</td>
                <td><b>{inf_time:.0f} ms</b> &nbsp;<span style="color:#8B949E;font-size:0.8rem;">{speed_note}</span></td>
            </tr>
            <tr>
                <td>Confidence Threshold</td>
                <td><b>{threshold:.2f}</b></td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================
if selected == "Home":

    st.title("🫀 Cardiomegaly Detection System")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown("""
        ### Selamat Datang 👋

        Sistem ini digunakan untuk mendeteksi **Kardiomegali**
        menggunakan citra **X-Ray Thoraks** berbasis Deep Learning.

        ### 🚀 Fitur Sistem
        - YOLOv11 Object Detection
        - SSD300 VGG16 Object Detection
        - Bounding Box Detection
        - CLAHE Enhancement
        - Real-time Inference
        - Confidence Score

        ### 📋 Cara Penggunaan
        1. Buka halaman **Detection**
        2. Upload gambar X-ray (JPG/PNG)
        3. Pilih model deteksi
        4. Atur Confidence Threshold
        5. Lihat hasil dan interpretasi deteksi

        ### ⚠️ Disclaimer
        Sistem ini hanya digunakan untuk:
        - Penelitian
        - Edukasi
        - Tugas akhir

        Bukan pengganti diagnosis dokter.
        """)

    with col2:

        st.markdown("""
        <div class="custom-card">
            <h3>📊 Quick Stats</h3>
        </div>
        """, unsafe_allow_html=True)

        st.metric("YOLOv11 Input", "512×512")
        st.metric("SSD300 Input", "300×300")
        st.metric("Inference", "< 2 sec")
        st.metric("Kelas", "Normal / Kardiomegali")


# ============================================================
# DETECTION
# ============================================================
elif selected == "Detection":

    st.title("🔬 X-Ray Detection")
    st.markdown("---")

    # -- Kontrol --
    col1, col2, col3 = st.columns(3)

    with col1:
        model_choice = st.selectbox(
            "Pilih Model",
            ["YOLOv11", "SSD300 (VGG16)", "Both Models"]
        )

    with col2:
        conf_thresh = st.slider(
            "Confidence Threshold",
            0.10, 0.90, 0.25, 0.05
        )

    with col3:
        run_norm = st.checkbox("CLAHE Enhancement", value=True)

    # -- Upload --
    uploaded = st.file_uploader(
        "Upload Gambar X-ray",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is None:
        st.warning("📌 Silakan upload gambar X-ray terlebih dahulu.")
        st.stop()

    # -- Load image --
    try:
        original = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("❌ File gambar tidak valid.")
        st.stop()

    # -- Validasi --
    with st.spinner("🔍 Memvalidasi X-ray..."):
        valid = is_valid_xray(original)

    if not valid:
        st.error("❌ Gambar bukan Chest X-ray yang valid.")
        st.stop()

    st.success("✅ Chest X-ray valid terdeteksi")

    # -- CLAHE --
    if run_norm:
        with st.spinner("⚙️ Menerapkan CLAHE Enhancement..."):
            processed = normalize_xray(original)
    else:
        processed = original

    # ── Preview Gambar ──────────────────────────────────────
    st.markdown("### 🖼️ Preview Gambar")

    prev_col1, prev_col2 = st.columns(2)

    with prev_col1:
        st.markdown("**📷 Original Image**")
        st.image(original, use_column_width=True)

    with prev_col2:
        label_clahe = "**✨ Setelah CLAHE Enhancement**" if run_norm else "**📷 Gambar (CLAHE Nonaktif)**"
        st.markdown(label_clahe)
        st.image(processed, use_column_width=True)

    st.markdown("---")

    # ────────────────────────────────────────────────────────
    # BOTH MODELS — berdampingan penuh
    # ────────────────────────────────────────────────────────
    if model_choice == "Both Models":

        st.markdown("### 🔍 Hasil Deteksi — Perbandingan Model")
        st.caption("Kedua model dijalankan pada gambar yang sama untuk memudahkan perbandingan.")

        col_yolo, col_ssd = st.columns(2)

        # ---------- YOLO ----------
        with col_yolo:
            st.markdown("""
            <div class="section-header">
                <b>🔵 YOLOv11s</b>
            </div>
            """, unsafe_allow_html=True)

            if not yolo_available:
                st.error("YOLO model tidak ditemukan.")
            else:
                with st.spinner("Menjalankan YOLOv11s..."):
                    start      = time.perf_counter()
                    yolo_img, yolo_dets = predict_yolo(
                        yolo_model, processed, conf_thresh
                    )
                    yolo_time  = (time.perf_counter() - start) * 1000

                # Gambar hasil
                st.markdown("**Hasil Bounding Box**")
                st.image(yolo_img, use_column_width=True)

                # Statistik
                render_stats_card("YOLOv11s", len(yolo_dets), yolo_time, conf_thresh)

                # Hasil deteksi
                if yolo_dets:
                    st.markdown("**Detail Deteksi**")
                    for det in yolo_dets:
                        render_result_card(det)
                else:
                    st.info("ℹ️ Tidak ada objek terdeteksi di atas threshold.")

        # ---------- SSD ----------
        with col_ssd:
            st.markdown("""
            <div class="section-header">
                <b>🟡 SSD300 (VGG16)</b>
            </div>
            """, unsafe_allow_html=True)

            if not ssd_available:
                st.error("SSD model tidak ditemukan.")
            else:
                with st.spinner("Menjalankan SSD300..."):
                    start     = time.perf_counter()
                    ssd_img, ssd_dets = predict_ssd(
                        ssd_model, processed, conf_thresh, DEVICE
                    )
                    ssd_time  = (time.perf_counter() - start) * 1000

                # Gambar hasil
                st.markdown("**Hasil Bounding Box**")
                st.image(ssd_img, use_column_width=True)

                # Statistik
                render_stats_card("SSD300 VGG16", len(ssd_dets), ssd_time, conf_thresh)

                # Hasil deteksi
                if ssd_dets:
                    st.markdown("**Detail Deteksi**")
                    for det in ssd_dets:
                        render_result_card(det)
                else:
                    st.info("ℹ️ Tidak ada objek terdeteksi di atas threshold.")

    # ────────────────────────────────────────────────────────
    # YOLO saja — gambar kiri, info kanan
    # ────────────────────────────────────────────────────────
    elif model_choice == "YOLOv11s":

        st.markdown("### 🔵 Hasil Deteksi — YOLOv11s")

        if not yolo_available:
            st.error("YOLO model tidak ditemukan.")
        else:
            with st.spinner("Menjalankan YOLOv11s..."):
                start     = time.perf_counter()
                yolo_img, yolo_dets = predict_yolo(
                    yolo_model, processed, conf_thresh
                )
                yolo_time = (time.perf_counter() - start) * 1000

            col_img, col_info = st.columns([1, 1])

            with col_img:
                st.markdown("**📸 Hasil Bounding Box**")
                st.image(yolo_img, use_column_width=True)

            with col_info:
                st.markdown("**📋 Informasi Deteksi**")
                render_stats_card("YOLOv11s", len(yolo_dets), yolo_time, conf_thresh)

                if yolo_dets:
                    st.markdown("**🔎 Detail Hasil**")
                    for det in yolo_dets:
                        render_result_card(det)
                else:
                    st.info("ℹ️ Tidak ada objek terdeteksi di atas threshold.")

    # ────────────────────────────────────────────────────────
    # SSD saja — gambar kiri, info kanan
    # ────────────────────────────────────────────────────────
    elif model_choice == "SSD300 (VGG16)":

        st.markdown("### 🟡 Hasil Deteksi — SSD300 (VGG16)")

        if not ssd_available:
            st.error("SSD model tidak ditemukan.")
        else:
            with st.spinner("Menjalankan SSD300..."):
                start    = time.perf_counter()
                ssd_img, ssd_dets = predict_ssd(
                    ssd_model, processed, conf_thresh, DEVICE
                )
                ssd_time = (time.perf_counter() - start) * 1000

            col_img, col_info = st.columns([1, 1])

            with col_img:
                st.markdown("**📸 Hasil Bounding Box**")
                st.image(ssd_img, use_column_width=True)

            with col_info:
                st.markdown("**📋 Informasi Deteksi**")
                render_stats_card("SSD300 VGG16", len(ssd_dets), ssd_time, conf_thresh)

                if ssd_dets:
                    st.markdown("**🔎 Detail Hasil**")
                    for det in ssd_dets:
                        render_result_card(det)
                else:
                    st.info("ℹ️ Tidak ada objek terdeteksi di atas threshold.")

    # -- Disclaimer --
    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** Hasil deteksi ini hanya untuk keperluan penelitian dan edukasi. "
        "Bukan pengganti diagnosis dokter. Konsultasikan hasil dengan tenaga medis profesional."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================
elif selected == "Model Comparison":

    st.title("📊 Model Comparison")
    st.markdown("---")

    st.markdown("### YOLOv11s vs SSD300 VGG16")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>🔵 YOLOv11s</h3>
            <table>
                <tr><td>Arsitektur</td><td><b>CSPDarknet</b></td></tr>
                <tr><td>Input Size</td><td><b>512×512</b></td></tr>
                <tr><td>Kecepatan</td><td><b>Sangat Cepat ⚡</b></td></tr>
                <tr><td>Keunggulan</td><td>Real-time, efisien</td></tr>
                <tr><td>mAP50</td><td><b>95%</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>🟡 SSD300 VGG16</h3>
            <table>
                <tr><td>Arsitektur</td><td><b>VGG16 Backbone</b></td></tr>
                <tr><td>Input Size</td><td><b>300×300</b></td></tr>
                <tr><td>Kecepatan</td><td><b>Cepat ⚡</b></td></tr>
                <tr><td>Keunggulan</td><td>Multi-scale anchor</td></tr>
                <tr><td>mAP50</td><td><b>91%</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Performance Metrics")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("YOLO mAP50", "95%", delta="Lebih Tinggi")

    with m2:
        st.metric("SSD mAP50", "91%")

    with m3:
        st.metric("Inference Speed", "< 2 sec")

    st.markdown("---")
    st.subheader("🏆 Rekomendasi Penggunaan")

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("""
        <div class="custom-card">
            <h4>Gunakan YOLOv11s jika:</h4>
            <ul>
                <li>Membutuhkan deteksi cepat</li>
                <li>Resource terbatas (CPU)</li>
                <li>Akurasi tinggi dibutuhkan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        st.markdown("""
        <div class="custom-card">
            <h4>Gunakan SSD300 jika:</h4>
            <ul>
                <li>Ingin baseline yang stabil</li>
                <li>Perbandingan arsitektur klasik</li>
                <li>Multi-scale detection penting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# ABOUT
# ============================================================
elif selected == "About":

    st.title("ℹ️ About System")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📖 Overview",
        "🧠 Architecture",
        "📊 Pipeline"
    ])

    with tab1:

        st.markdown("""
        ### CardioDetect

        Sistem Object Detection berbasis Deep Learning
        untuk mendeteksi Kardiomegali pada
        citra X-ray Thoraks.

        ### Framework
        - PyTorch
        - Streamlit
        - Ultralytics YOLO
        - Torchvision SSD

        ### Dataset
        - Chest X-ray (Normal & Kardiomegali)
        - Augmentasi: Flip, Rotate, CLAHE
        """)

    with tab2:

        st.markdown("""
        ### YOLOv11s
        - CSPDarknet Backbone
        - Real-time detector
        - Faster inference
        - Input: 512×512

        ### SSD300
        - VGG16 Backbone
        - Multi-scale feature maps
        - Anchor box detection
        - Input: 300×300
        """)

    with tab3:

        st.markdown("""
        ### Detection Pipeline

        1. **Upload Image** — Format JPG/PNG
        2. **Validate X-ray** — Cek validitas citra thoraks
        3. **CLAHE Enhancement** — Perbaiki kontras gambar
        4. **Resize Image** — Sesuaikan ukuran input model
        5. **Model Inference** — Jalankan deteksi
        6. **Bounding Box** — Tampilkan kotak deteksi
        7. **Confidence Filtering** — Filter sesuai threshold
        8. **Interpretasi** — Tampilkan hasil + keterangan
        """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown("""
<div style='text-align:center; color:gray;'>
    🫀 CardioDetect — Deep Learning Thorax X-ray Detection <br>
    Tugas Akhir 2025
</div>
""", unsafe_allow_html=True)