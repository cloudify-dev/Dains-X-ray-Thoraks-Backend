"""
app.py — CardioDetect
Deteksi Kardiomegali X-Ray Thoraks
YOLOv11s + SSD300 VGG16
"""

from export_pdf import export_detection_pdf
from styles     import inject_styles, section_header

import time
import torch
import streamlit as st
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from streamlit_option_menu import option_menu

from styles import inject_styles


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

inject_styles()



# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_yolo_model(path: str):
    """
    Load YOLOv11 model menggunakan Ultralytics YOLO.
    YOLOv11 menggunakan API yang sama, cukup pastikan
    checkpoint adalah file yolo11*.pt hasil training.
    """
    return YOLO(path)


@st.cache_resource
def load_ssd_model(path: str, device: str):
    """
    Load SSD300 VGG16 model.
    Arsitektur sesuai notebook:
      - NUM_CLASSES = 3 (background, kardiomegali, normal)
      - in_channels = [512, 1024, 512, 256, 256, 256]
      - Checkpoint: outputs/ssd_best.pt
    """
    from torchvision.models.detection import ssd300_vgg16
    from torchvision.models.detection.ssd import SSDClassificationHead

    NUM_CLASSES = 3  # 0=background, 1=kardiomegali, 2=normal

    model = ssd300_vgg16(
        weights=None,
        score_thresh=0.01,
        nms_thresh=0.45,
        detections_per_img=10
    )

    in_channels  = [512, 1024, 512, 256, 256, 256]
    num_anchors  = model.anchor_generator.num_anchors_per_location()

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
# YOLOv11: sesuai notebook → outputs/yolo11/thorax/weights/best.pt
# Fallback tambahan untuk berbagai kemungkinan lokasi output
# SSD300  : outputs/ssd_best.pt
# ============================================================
def _find_yolo11_weight() -> str:
    """
    Mencari best.pt YOLOv11 dari beberapa kemungkinan lokasi,
    sesuai dengan logika find_yolo_best_pt() di notebook.
    """
    candidates = [
        Path('outputs') / 'yolo11' / 'thorax' / 'weights' / 'best.pt',
        Path('runs') / 'detect' / 'outputs' / 'yolo11' / 'thorax' / 'weights' / 'best.pt',
        Path('runs') / 'detect' / 'thorax' / 'weights' / 'best.pt',
        # legacy path (jika ada sisa folder YOLOv8 sebelumnya)
        Path('runs') / 'detect' / 'outputs' / 'yolov8' / 'thorax' / 'weights' / 'best.pt',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Glob fallback
    matches = list(Path('.').rglob('yolo11/thorax/weights/best.pt'))
    if matches:
        return str(matches[0])
    return str(candidates[0])   # kembalikan path utama meski belum ada


YOLO_WEIGHT = _find_yolo11_weight()
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
            X-ray Thoraks Cardiomegaly Object Detection
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
        st.success("YOLOv11s Loaded ✅")
    else:
        st.error("YOLOv11s Not Found ❌")
        st.caption(f"Expected: `{YOLO_WEIGHT}`")

    if ssd_available:
        st.success("SSD300 Loaded ✅")
    else:
        st.error("SSD300 Not Found ❌")
        st.caption(f"Expected: `{SSD_WEIGHT}`")

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
        - YOLOv11s Object Detection
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

        st.metric("YOLOv11s Input", "512×512")
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
            ["YOLOv11s", "SSD300 (VGG16)", "Both Models"]
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
                st.error("YOLOv11s model tidak ditemukan.")
                st.caption(f"Path: `{YOLO_WEIGHT}`")
            else:
                with st.spinner("Menjalankan YOLOv11s..."):
                    start      = time.perf_counter()
                    yolo_img, yolo_dets = predict_yolo(
                        yolo_model, processed, conf_thresh
                    )
                    yolo_time  = (time.perf_counter() - start) * 1000

                st.markdown("**Hasil Bounding Box**")
                st.image(yolo_img, use_column_width=True)

                render_stats_card("YOLOv11s", len(yolo_dets), yolo_time, conf_thresh)

                if yolo_dets:
                    st.markdown("**Detail Deteksi**")
                    for det in yolo_dets:
                        render_result_card(det)
                else:
                    st.info("ℹ️ Tidak ada objek terdeteksi di atas threshold.")
                    
            pdf_yolo = export_detection_pdf(
            model_name="YOLOv11s",
            original_img=original,
            result_img=yolo_img,
            detections=yolo_dets,
            inf_time_ms=yolo_time,
            conf_thresh=conf_thresh,
            clahe_applied=run_norm,
        )

        st.download_button(
            label="📥 Download YOLO PDF",
            data=pdf_yolo,
            file_name="cardiodetect_yolo_report.pdf",
            mime="application/pdf",
            key="yolo_pdf"
        )

        # ---------- SSD ----------
        with col_ssd:
            st.markdown("""
            <div class="section-header">
                <b>🟡 SSD300 (VGG16)</b>
            </div>
            """, unsafe_allow_html=True)

            if not ssd_available:
                st.error("SSD model tidak ditemukan.")
                st.caption(f"Path: `{SSD_WEIGHT}`")
            else:
                with st.spinner("Menjalankan SSD300..."):
                    start     = time.perf_counter()
                    ssd_img, ssd_dets = predict_ssd(
                        ssd_model, processed, conf_thresh, DEVICE
                    )
                    ssd_time  = (time.perf_counter() - start) * 1000

                st.markdown("**Hasil Bounding Box**")
                st.image(ssd_img, use_column_width=True)

                render_stats_card("SSD300 VGG16", len(ssd_dets), ssd_time, conf_thresh)

                if ssd_dets:
                    st.markdown("**Detail Deteksi**")
                    for det in ssd_dets:
                        render_result_card(det)
                else:
                    st.info("ℹ️ Tidak ada objek terdeteksi di atas threshold.")

    # # ────────────────────────────────────────────────────────
    # # YOLO saja — gambar kiri, info kanan
    # # ────────────────────────────────────────────────────────
    # elif model_choice == "YOLOv11s":

    #     st.markdown("### 🔵 Hasil Deteksi — YOLOv11s")

    #     if not yolo_available:
    #         st.error("YOLOv11s model tidak ditemukan.")
    #         st.caption(f"Path: `{YOLO_WEIGHT}`")
    #     else:
    #         with st.spinner("Menjalankan YOLOv11s..."):
    #             start     = time.perf_counter()
    #             yolo_img, yolo_dets = predict_yolo(
    #                 yolo_model, processed, conf_thresh
    #             )
    #             yolo_time = (time.perf_counter() - start) * 1000

    #         col_img, col_info = st.columns([1, 1])

    #         with col_img:
    #             st.markdown("**📸 Hasil Bounding Box**")
    #             st.image(yolo_img, use_column_width=True)

    #         with col_info:
    #             st.markdown("**📋 Informasi Deteksi**")
    #             render_stats_card("YOLOv11s", len(yolo_dets), yolo_time, conf_thresh)

    #             if yolo_dets:
    #                 st.markdown("**🔎 Detail Hasil**")
    #                 for det in yolo_dets:
    #                     render_result_card(det)
    #             else:
    #                 st.info("ℹ️ Tidak ada objek terdeteksi di atas threshold.")
    
    # ============================================================
    # YOLO saja — gambar kiri, info kanan
    # ============================================================
    elif model_choice == "YOLOv11s":

        st.markdown("### 🔵 Hasil Deteksi — YOLOv11s")

        if not yolo_available:
            st.error("YOLOv11s model tidak ditemukan.")
            st.caption(f"Path: `{YOLO_WEIGHT}`")
        else:
            with st.spinner("Menjalankan YOLOv11s..."):
                start     = time.perf_counter()

                yolo_img, yolo_dets = predict_yolo(
                    yolo_model,
                    processed,
                    conf_thresh
                )

                yolo_time = (time.perf_counter() - start) * 1000

            col_img, col_info = st.columns([1, 1])

        # ====================================================
        # IMAGE RESULT
        # ====================================================
        with col_img:

            st.markdown("**📸 Hasil Bounding Box**")

            st.image(
                yolo_img,
                use_column_width=True
            )

        # ====================================================
        # DETECTION INFO
        # ====================================================
        with col_info:

            st.markdown("**📋 Informasi Deteksi**")

            render_stats_card(
                "YOLOv11s",
                len(yolo_dets),
                yolo_time,
                conf_thresh
            )

            if yolo_dets:

                st.markdown("**🔎 Detail Hasil**")

                for det in yolo_dets:
                    render_result_card(det)

            else:
                st.info(
                    "ℹ️ Tidak ada objek terdeteksi "
                    "di atas threshold."
                )

            # =================================================
            # EXPORT PDF
            # =================================================
            st.markdown("---")
            st.markdown("### 📄 Export Report")

            try:

                pdf_bytes = export_detection_pdf(
                    model_name="YOLOv11s",
                    original_img=original,
                    result_img=yolo_img,
                    detections=yolo_dets,
                    inf_time_ms=yolo_time,
                    conf_thresh=conf_thresh,
                    clahe_applied=run_norm,
                )

                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="cardiodetect_yolov11_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                st.success("✅ PDF report siap di-download")

            except Exception as e:

                st.error(
                    f"❌ Gagal membuat PDF report:\n{e}"
                )

    # # ────────────────────────────────────────────────────────
    # # SSD saja — gambar kiri, info kanan
    # # ────────────────────────────────────────────────────────
    # elif model_choice == "SSD300 (VGG16)":

    #     st.markdown("### 🟡 Hasil Deteksi — SSD300 (VGG16)")

    #     if not ssd_available:
    #         st.error("SSD model tidak ditemukan.")
    #         st.caption(f"Path: `{SSD_WEIGHT}`")
    #     else:
    #         with st.spinner("Menjalankan SSD300..."):
    #             start    = time.perf_counter()
    #             ssd_img, ssd_dets = predict_ssd(
    #                 ssd_model, processed, conf_thresh, DEVICE
    #             )
    #             ssd_time = (time.perf_counter() - start) * 1000

    #         col_img, col_info = st.columns([1, 1])

    #         with col_img:
    #             st.markdown("**📸 Hasil Bounding Box**")
    #             st.image(ssd_img, use_column_width=True)

    #         with col_info:
    #             st.markdown("**📋 Informasi Deteksi**")
    #             render_stats_card("SSD300 VGG16", len(ssd_dets), ssd_time, conf_thresh)

    #             if ssd_dets:
    #                 st.markdown("**🔎 Detail Hasil**")
    #                 for det in ssd_dets:
    #                     render_result_card(det)
    #             else:
    #                 st.info("ℹ️ Tidak ada objek terdeteksi di atas threshold.")
    
    # ============================================================
    # SSD saja — gambar kiri, info kanan
    # ============================================================
    elif model_choice == "SSD300 (VGG16)":

        st.markdown("### 🟡 Hasil Deteksi — SSD300 (VGG16)")

        if not ssd_available:
            st.error("SSD model tidak ditemukan.")
            st.caption(f"Path: `{SSD_WEIGHT}`")
        else:
            with st.spinner("Menjalankan SSD300..."):

                start = time.perf_counter()

                ssd_img, ssd_dets = predict_ssd(
                    ssd_model,
                    processed,
                    conf_thresh,
                    DEVICE
                )

                ssd_time = (
                    time.perf_counter() - start
                ) * 1000

            col_img, col_info = st.columns([1, 1])

            # ====================================================
            # IMAGE RESULT
            # ====================================================
            with col_img:

                st.markdown("**📸 Hasil Bounding Box**")

                st.image(
                    ssd_img,
                    use_column_width=True
                )

            # ====================================================
            # DETECTION INFO
            # ====================================================
            with col_info:

                st.markdown("**📋 Informasi Deteksi**")

                render_stats_card(
                    "SSD300 VGG16",
                    len(ssd_dets),
                    ssd_time,
                    conf_thresh
                )

                if ssd_dets:

                    st.markdown("**🔎 Detail Hasil**")

                    for det in ssd_dets:
                        render_result_card(det)

                else:

                    st.info(
                        "ℹ️ Tidak ada objek terdeteksi "
                        "di atas threshold."
                    )

                # =================================================
                # EXPORT PDF
                # =================================================
                st.markdown("---")
                st.markdown("### 📄 Export Report")

                try:

                    pdf_bytes = export_detection_pdf(
                        model_name="SSD300 VGG16",
                        original_img=original,
                        result_img=ssd_img,
                        detections=ssd_dets,
                        inf_time_ms=ssd_time,
                        conf_thresh=conf_thresh,
                        clahe_applied=run_norm,
                    )

                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name="cardiodetect_ssd300_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

                    st.success(
                        "✅ PDF report siap di-download"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Gagal membuat PDF report:\n{e}"
                    )

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
                <tr><td>Arsitektur</td><td><b>C3k2 + C2PSA</b></td></tr>
                <tr><td>Backbone</td><td><b>CSPDarknet v11</b></td></tr>
                <tr><td>Input Size</td><td><b>512×512</b></td></tr>
                <tr><td>Kecepatan</td><td><b>Sangat Cepat ⚡</b></td></tr>
                <tr><td>Keunggulan</td><td>Real-time, efisien, attention</td></tr>
                <tr><td>Training Epochs</td><td><b>25 (early stop)</b></td></tr>
                <tr><td>Optimizer</td><td><b>SGD</b></td></tr>
                <tr><td>mAP50</td><td><b>95%</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>🟡 SSD300 VGG16</h3>
            <table>
                <tr><td>Arsitektur</td><td><b>SSD300</b></td></tr>
                <tr><td>Backbone</td><td><b>VGG16</b></td></tr>
                <tr><td>Input Size</td><td><b>300×300</b></td></tr>
                <tr><td>Kecepatan</td><td><b>Cepat ⚡</b></td></tr>
                <tr><td>Keunggulan</td><td>Multi-scale anchor</td></tr>
                <tr><td>Training Epochs</td><td><b>25 (early stop)</b></td></tr>
                <tr><td>Optimizer</td><td><b>SGD + Nesterov</b></td></tr>
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
    st.subheader("Rekomendasi Penggunaan")

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("""
        <div class="custom-card">
            <h4>Gunakan YOLOv11s jika:</h4>
            <ul>
                <li>Membutuhkan deteksi cepat & akurat</li>
                <li>Resource terbatas (CPU)</li>
                <li>Ingin memanfaatkan attention mechanism</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        st.markdown("""
        <div class="custom-card">
            <h4>Gunakan SSD300 jika:</h4>
            <ul>
                <li>Ingin baseline arsitektur klasik</li>
                <li>Perbandingan arsitektur konvensional</li>
                <li>Multi-scale detection penting</li>
                <li>Akurasi tinggi dibutuhkan</li>
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
        ### Cardiomegaly Object Detection

        Sistem Object Detection berbasis Deep Learning
        untuk mendeteksi Kardiomegali pada
        citra X-ray Thoraks.

        ### Framework
        - PyTorch
        - Streamlit
        - Ultralytics YOLO (v11)
        - Torchvision SSD

        ### Dataset
        - Chest X-ray (Normal & Kardiomegali)
        - Augmentasi: Rotate ±3°, Translate ±5%, Scale ±10%, CLAHE, HSV-V
        """)

    with tab2:

        st.markdown("""
        ### YOLOv11s
        - Backbone: CSPDarknet v11 dengan C3k2 + C2PSA (attention)
        - Head: Decoupled detection head
        - Real-time detector, faster & more accurate vs YOLOv8
        - Input: 512×512
        - Dilatih 25 epoch dengan SGD + early stopping (patience=7)

        ### SSD300
        - Backbone: VGG16 (pretrained features)
        - Multi-scale feature maps untuk berbagai ukuran objek
        - Anchor box-based detection
        - Input: 300×300
        - NUM_CLASSES: 3 (background, kardiomegali, normal)
        - Dilatih 25 epoch dengan SGD Nesterov + MultiStepLR
        """)

    with tab3:

        st.markdown("""
        ### Detection Pipeline

        1. **Upload Image** — Format JPG/PNG
        2. **Validate X-ray** — Cek validitas citra thoraks (histogram)
        3. **CLAHE Enhancement** — Perbaiki kontras (clipLimit=2.0, tile=8×8)
        4. **Resize Image** — YOLOv11s: 512×512 | SSD: 300×300
        5. **Model Inference** — Jalankan deteksi
        6. **Bounding Box** — Tampilkan kotak deteksi + label
        7. **Confidence Filtering** — Filter sesuai threshold
        8. **Interpretasi** — Tampilkan hasil + keterangan klinis
        """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown("""
<div style='text-align:center; color:gray;'>
    🫀 Cardiomegaly Object Detection | Deep Learning, Computer Vision<br>
    YOLOv11s, SSD300 VGG16 | Tugas Akhir 2026
</div>
""", unsafe_allow_html=True)