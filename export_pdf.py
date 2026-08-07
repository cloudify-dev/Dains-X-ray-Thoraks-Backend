"""
export_pdf.py — CardioDetect
Laporan hasil deteksi kardiomegali: putih bersih, simpel, elegan.
Hanya informasi penting yang ditampilkan.
"""

import io
from datetime import datetime
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether,
)

# ── Palet warna  ─────────────────────────────────────────────────────────────
C_NAVY      = colors.HexColor('#1E3A5F')   # header teks, judul
C_BLUE      = colors.HexColor('#2563EB')   # aksen, header tabel
C_BLUE_SOFT = colors.HexColor('#DBEAFE')   # grid tabel
C_BLUE_ROW  = colors.HexColor('#F0F7FF')   # baris tabel alt
C_GREEN     = colors.HexColor('#16A34A')   # hasil normal
C_GREEN_BG  = colors.HexColor('#F0FDF4')   # bg hasil normal
C_GREEN_BOR = colors.HexColor('#BBF7D0')   # border normal
C_RED       = colors.HexColor('#DC2626')   # hasil kardiomegali
C_RED_BG    = colors.HexColor('#FEF2F2')   # bg kardiomegali
C_RED_BOR   = colors.HexColor('#FECACA')   # border kardiomegali
C_GRAY      = colors.HexColor('#6B7280')   # teks sekunder
C_LIGHT     = colors.HexColor('#F9FAFB')   # bg card abu tipis
C_BORDER    = colors.HexColor('#E5E7EB')   # garis halus
C_WHITE     = colors.white
C_BLACK     = colors.HexColor('#111827')

PAGE_W, PAGE_H = A4
M = 18 * mm                               # margin
USABLE_W = PAGE_W - 2 * M


# ── Style helpers ─────────────────────────────────────────────────────────────
def _s(name, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)

ST = {
    'report_title': _s('rt',
        fontName='Helvetica-Bold', fontSize=17,
        textColor=C_NAVY, alignment=TA_CENTER, spaceAfter=2),
    'report_sub': _s('rs',
        fontName='Helvetica', fontSize=9,
        textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=2),
    'section': _s('sec',
        fontName='Helvetica-Bold', fontSize=10,
        textColor=C_NAVY, spaceBefore=8, spaceAfter=5),
    'body': _s('bod',
        fontName='Helvetica', fontSize=9,
        textColor=C_BLACK, leading=14, spaceAfter=3),
    'label': _s('lbl',
        fontName='Helvetica', fontSize=9, textColor=C_GRAY),
    'value': _s('val',
        fontName='Helvetica-Bold', fontSize=9, textColor=C_BLACK),
    'result_ok': _s('rok',
        fontName='Helvetica-Bold', fontSize=14,
        textColor=C_GREEN, alignment=TA_CENTER),
    'result_bad': _s('rbd',
        fontName='Helvetica-Bold', fontSize=14,
        textColor=C_RED, alignment=TA_CENTER),
    'result_sub': _s('rsb',
        fontName='Helvetica', fontSize=8,
        textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=0),
    'disclaimer': _s('dis',
        fontName='Helvetica-Oblique', fontSize=7.5,
        textColor=C_GRAY, alignment=TA_CENTER, leading=11),
    'footer': _s('ftr',
        fontName='Helvetica', fontSize=7,
        textColor=C_GRAY, alignment=TA_CENTER),
    'img_label': _s('iml',
        fontName='Helvetica-Bold', fontSize=8,
        textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=2),
}


# ── Konversi PIL → ReportLab Image ───────────────────────────────────────────
def _rl_img(pil_img: Image.Image, max_w: float, max_h: float) -> RLImage:
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    ow, oh = pil_img.size
    r = min(max_w / ow, max_h / oh)
    return RLImage(buf, width=ow * r, height=oh * r)


# ── Logo placeholder (teks) ──────────────────────────────────────────────────
def _logo_left() -> str:
    return "<b><font color='#1E3A5F' size=11>♥ CardioDetect</font></b>"

def _logo_right() -> str:
    return "<font color='#6B7280' size=8>Cardiomegaly Detection System</font>"


# ── Header/Footer setiap halaman ─────────────────────────────────────────────
class _PageDeco:
    def __init__(self, timestamp: str):
        self.ts = timestamp

    def __call__(self, canv, doc):
        canv.saveState()
        w, h = A4

        # ── Header bar tipis biru ──────────────────────────
        canv.setFillColor(C_NAVY)
        canv.rect(0, h - 16*mm, w, 16*mm, fill=1, stroke=0)

        # Aksen garis biru terang di bawah header
        canv.setFillColor(C_BLUE)
        canv.rect(0, h - 17*mm, w, 1*mm, fill=1, stroke=0)

        # Teks kiri: logo
        canv.setFont('Helvetica-Bold', 11)
        canv.setFillColor(C_WHITE)
        canv.drawString(M, h - 10*mm, '\u2665 CardioDetect')

        # Teks kanan: subtitle
        canv.setFont('Helvetica', 8)
        canv.setFillColor(colors.HexColor('#93C5FD'))
        canv.drawRightString(w - M, h - 10*mm,
                             'Cardiomegaly Detection Report')

        # ── Garis bawah tipis (pemisah konten) ────────────
        canv.setStrokeColor(C_BORDER)
        canv.setLineWidth(0.5)
        canv.line(M, h - 19*mm, w - M, h - 19*mm)

        # ── Footer ────────────────────────────────────────
        canv.setStrokeColor(C_BORDER)
        canv.line(M, 12*mm, w - M, 12*mm)

        canv.setFont('Helvetica', 7)
        canv.setFillColor(C_GRAY)
        canv.drawString(M, 8*mm,
            f'Dibuat: {self.ts}  \u2022  CardioDetect \u2014 Tugas Akhir 2026')
        canv.drawRightString(w - M, 8*mm, f'Halaman {doc.page}')

        canv.restoreState()


# ── Tabel dua kolom: label | nilai ───────────────────────────────────────────
def _kv_table(rows: list[tuple],
              col_w=None,
              header_row: tuple | None = None) -> Table:
    """
    rows       : [(label, value), ...]
    header_row : tuple (col1, col2) untuk baris header berwarna biru — opsional
    """
    cw = col_w or [USABLE_W * 0.40, USABLE_W * 0.60]

    data = []
    if header_row:
        data.append([
            Paragraph(f'<b>{header_row[0]}</b>',
                      ParagraphStyle('hd', fontName='Helvetica-Bold',
                                     fontSize=9, textColor=C_WHITE)),
            Paragraph(f'<b>{header_row[1]}</b>',
                      ParagraphStyle('hd2', fontName='Helvetica-Bold',
                                     fontSize=9, textColor=C_WHITE)),
        ])

    for lbl, val in rows:
        data.append([
            Paragraph(lbl, ST['label']),
            Paragraph(val, ST['value']),
        ])

    tbl = Table(data, colWidths=cw)

    style_cmds = [
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('TOPPADDING',   (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
        ('LEFTPADDING',  (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('GRID',         (0, 0), (-1, -1), 0.4, C_BLUE_SOFT),
        ('BOX',          (0, 0), (-1, -1), 0.6, C_BORDER),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [C_WHITE, C_BLUE_ROW] if not header_row
         else [C_BLUE] + [C_WHITE, C_BLUE_ROW] * len(rows)),
    ]
    if header_row:
        style_cmds += [
            ('BACKGROUND',   (0, 0), (-1, 0), C_BLUE),
            ('TEXTCOLOR',    (0, 0), (-1, 0), C_WHITE),
            ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ── Banner hasil utama ────────────────────────────────────────────────────────
def _result_banner(det: dict) -> Table:
    cls  = det['class_name'].lower()
    conf = det['confidence'] * 100

    is_cardio = cls == 'kardiomegali'
    icon      = '✖  KARDIOMEGALI TERDETEKSI' if is_cardio else '✔  NORMAL'
    bg        = C_RED_BG    if is_cardio else C_GREEN_BG
    border_c  = C_RED       if is_cardio else C_GREEN
    txt_style = ST['result_bad'] if is_cardio else ST['result_ok']
    note      = ('Segera konsultasikan dengan dokter spesialis jantung.'
                 if is_cardio else
                 'Tidak ditemukan indikasi kardiomegali.')

    data = [
        [Paragraph(icon, txt_style)],
        [Paragraph(f'Confidence: <b>{conf:.1f}%</b>', ST['result_sub'])],
        [Paragraph(note, ST['result_sub'])],
    ]
    tbl = Table(data, colWidths=[USABLE_W])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), bg),
        ('BOX',           (0, 0), (-1, -1), 1.2, border_c),
        ('LINEAFTER',     (0, 0), (0, -1),  5,   border_c),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING',   (0, 0), (-1, -1), 18),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 14),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
    ]))
    return tbl


# ── FUNGSI UTAMA ──────────────────────────────────────────────────────────────
def export_detection_pdf(
    model_name   : str,
    original_img : Image.Image,
    result_img   : Image.Image,
    detections   : list[dict],
    inf_time_ms  : float,
    conf_thresh  : float,
    clahe_applied: bool = True,
    patient_name : str = "",
    patient_dob  : str = "",
) -> bytes:
    """
    Generate laporan PDF hasil deteksi kardiomegali.

    Parameters
    ----------
    model_name    : nama model yang digunakan ("YOLOv11s" / "SSD300 VGG16")
    original_img  : citra X-ray asli (PIL Image)
    result_img    : citra hasil deteksi dengan bounding box (PIL Image)
    detections    : list of dict hasil deteksi
    inf_time_ms   : waktu inferensi dalam milidetik
    conf_thresh   : confidence threshold yang digunakan
    clahe_applied : status CLAHE enhancement (True/False)
    patient_name  : nama pasien (opsional, default kosong)
    patient_dob   : tanggal lahir pasien dalam bentuk string (opsional, default kosong)

    Returns
    -------
    bytes — konten PDF siap di-download lewat st.download_button
    """
    buf       = io.BytesIO()
    now       = datetime.now()
    ts        = now.strftime('%d %B %Y  %H:%M')
    date_str  = now.strftime('%d %B %Y')
    time_str  = now.strftime('%H:%M:%S')

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=24*mm, bottomMargin=20*mm,
        leftMargin=M, rightMargin=M,
    )
    page_cb = _PageDeco(ts)
    story   = []

    # ── Judul ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph('LAPORAN HASIL DETEKSI', ST['report_title']))
    story.append(Spacer(2, 5*mm))
    story.append(HRFlowable(width=USABLE_W, thickness=1,
                             color=C_BLUE, spaceAfter=6))

    # ── DATA PASIEN ───────────────────────────────────────────────────────────
    story.append(Paragraph('Data Pasien', ST['section']))

    patient_rows = [
        ('Nama Pasien',   patient_name.strip() if patient_name and patient_name.strip() else '-'),
        ('Tanggal Lahir', patient_dob.strip() if patient_dob and patient_dob.strip() else '-'),
        ('Tanggal Pemeriksaan', date_str),
    ]
    story.append(KeepTogether([
        _kv_table(patient_rows, header_row=('Identitas', 'Keterangan'))
    ]))
    story.append(Spacer(1, 4*mm))

    # ── HASIL UTAMA (banner besar) ────────────────────────────────────────────
    story.append(Paragraph('Hasil Deteksi', ST['section']))

    if detections:
        for det in detections:
            story.append(_result_banner(det))
            story.append(Spacer(1, 2*mm))
    else:
        no_det = Table(
            [[Paragraph('— Tidak ada objek terdeteksi di atas threshold —',
                        ST['result_sub'])]],
            colWidths=[USABLE_W]
        )
        no_det.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), C_LIGHT),
            ('BOX',           (0,0),(-1,-1), 0.6, C_BORDER),
            ('TOPPADDING',    (0,0),(-1,-1), 12),
            ('BOTTOMPADDING', (0,0),(-1,-1), 12),
            ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ]))
        story.append(no_det)

    story.append(Spacer(1, 4*mm))

    # ── GAMBAR (original kiri | bounding box kanan) ───────────────────────────
    story.append(Paragraph('Citra X-Ray', ST['section']))

    img_w  = (USABLE_W - 6*mm) / 2
    img_h  = 60*mm

    rl_orig   = _rl_img(original_img, img_w, img_h)
    rl_result = _rl_img(result_img,   img_w, img_h)

    img_tbl = Table(
        [
            [Paragraph('Original Image',    ST['img_label']),
             Paragraph('Hasil Bounding Box', ST['img_label'])],
            [rl_orig, rl_result],
        ],
        colWidths=[img_w, img_w],
        hAlign='CENTER',
    )
    img_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), C_LIGHT),
        ('BOX',           (0,0),(-1,-1), 0.6, C_BORDER),
        ('INNERGRID',     (0,0),(-1,-1), 0.4, C_BORDER),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0),(-1,-1), 6),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('LEFTPADDING',   (0,0),(-1,-1), 6),
        ('RIGHTPADDING',  (0,0),(-1,-1), 6),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 5*mm))
    
    story.append(Spacer(1, 50*mm))
    # ── INFO RINGKAS (satu tabel beberapa baris) ──────────────────────────────
    story.append(Paragraph('Informasi Pemeriksaan', ST['section']))

    speed = ('Cepat' if inf_time_ms < 500 else
             'Normal' if inf_time_ms < 1500 else 'Lambat')
    conf_pct = f'{conf_thresh * 100:.0f}%'

    info_rows = [
        ('Tanggal &amp; Waktu',    f'{date_str}  {time_str}'),
        ('Model Deteksi',          f'<b>{model_name}</b>'),
        ('Confidence Threshold',   conf_pct),
        ('Waktu Inferensi',        f'{inf_time_ms:.0f} ms ({speed})'),
        ('CLAHE Enhancement',      'Aktif' if clahe_applied else 'Nonaktif'),
        ('Jumlah Deteksi',         f'{len(detections)} objek'),
    ]
    story.append(KeepTogether([
        _kv_table(info_rows, header_row=('Parameter', 'Nilai'))
    ]))
    story.append(Spacer(1, 5*mm))

    # ── DISCLAIMER singkat ───────────────────────────────────────────────────
    story.append(HRFlowable(width=USABLE_W, thickness=0.5,
                             color=C_BORDER, spaceBefore=2, spaceAfter=6))

    disc_tbl = Table(
        [[Paragraph(
            '<b>Disclaimer:</b> Laporan ini bersifat akademis dan tidak menggantikan '
            'diagnosis dokter. Selalu konsultasikan hasil dengan tenaga medis profesional.',
            ST['disclaimer']
        )]],
        colWidths=[USABLE_W]
    )
    disc_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), C_LIGHT),
        ('BOX',           (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 8),
        ('BOTTOMPADDING', (0,0),(-1,-1), 8),
        ('LEFTPADDING',   (0,0),(-1,-1), 12),
        ('RIGHTPADDING',  (0,0),(-1,-1), 12),
    ]))
    story.append(disc_tbl)

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=page_cb, onLaterPages=page_cb)
    buf.seek(0)
    return buf.read()