"""
styles.py — CardioDetect
Semua CSS dan fungsi inject styling dipisahkan ke sini
supaya app.py tetap bersih dan hanya berisi logika bisnis.

Cara pemakaian di app.py:
    from styles import inject_styles
    inject_styles()
"""

import streamlit as st

# ============================================================
# CSS UTAMA  (dark-theme CardioDetect)
# ============================================================
_MAIN_CSS = """
<style>

/* ── Reset & Font ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ── Sembunyikan elemen bawaan Streamlit ───────────────── */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }

/* ── App Background ────────────────────────────────────── */
.stApp {
    background-color: #0E1117;
    color: #C9D1D9;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}

/* ── Card Komponen ──────────────────────────────────────── */
.cd-card {
    background-color : #161B22;
    padding          : 20px 24px;
    border-radius    : 14px;
    border           : 1px solid #30363D;
    margin-bottom    : 20px;
    transition       : border-color 0.2s ease;
}
.cd-card:hover {
    border-color: #58A6FF;
}

/* ── Section Header ─────────────────────────────────────── */
.cd-section-header {
    background    : linear-gradient(135deg, #161B22, #1C2128);
    border        : 1px solid #30363D;
    border-radius : 10px;
    padding       : 12px 18px;
    margin-bottom : 14px;
    font-weight   : 600;
    font-size     : 0.95rem;
}

/* ── Result Boxes ───────────────────────────────────────── */
.cd-result-box {
    padding       : 18px 20px;
    border-radius : 12px;
    margin-top    : 12px;
    border        : 1px solid #30363D;
}
.cd-result-normal {
    background-color : rgba(46, 160, 67, 0.12);
    border-left      : 4px solid #2EA043 !important;
}
.cd-result-cardio {
    background-color : rgba(248, 81, 73, 0.12);
    border-left      : 4px solid #F85149 !important;
}

/* ── Info Table di dalam result-box ─────────────────────── */
.cd-result-box table {
    width           : 100%;
    margin-top      : 10px;
    font-size       : 0.88rem;
    border-collapse : collapse;
}
.cd-result-box table tr td {
    padding     : 5px 4px;
    vertical-align: top;
}
.cd-result-box table tr td:first-child {
    color : #8B949E;
    width : 42%;
}

/* ── Stats Table di dalam cd-card ───────────────────────── */
.cd-card table {
    width           : 100%;
    border-collapse : collapse;
    font-size       : 0.88rem;
}
.cd-card table tr td {
    padding       : 7px 4px;
    border-bottom : 1px solid #21262D;
}
.cd-card table tr:last-child td {
    border-bottom: none;
}
.cd-card table tr td:first-child {
    color : #8B949E;
    width : 55%;
}

/* ── Metric Cards ───────────────────────────────────────── */
.cd-metric {
    background    : #161B22;
    border        : 1px solid #30363D;
    border-radius : 12px;
    padding       : 16px;
    text-align    : center;
    transition    : transform 0.15s ease, border-color 0.2s ease;
}
.cd-metric:hover {
    transform    : translateY(-2px);
    border-color : #58A6FF;
}
.cd-metric .value {
    font-size   : 1.6rem;
    font-weight : 700;
    color       : #58A6FF;
}
.cd-metric .label {
    font-size : 0.8rem;
    color     : #8B949E;
    margin-top: 4px;
}

/* ── Upload Area ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color : #161B22;
    border           : 2px dashed #30363D;
    border-radius    : 10px;
    padding          : 10px;
    transition       : border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #58A6FF;
}

/* ── Tombol ─────────────────────────────────────────────── */
/* Tombol Deteksi (primary) */
.stButton > button {
    background-color : #238636;
    color            : white;
    border-radius    : 10px;
    border           : 1px solid #2EA043;
    font-weight      : 600;
    transition       : background-color 0.2s ease, transform 0.1s ease;
}
.stButton > button:hover {
    background-color : #2EA043;
    transform        : translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* Tombol Download PDF — kelas khusus via HTML trick */
.cd-btn-pdf > button {
    background-color : #1F3B5A !important;
    color            : #58A6FF !important;
    border           : 1px solid #58A6FF !important;
    border-radius    : 10px !important;
}
.cd-btn-pdf > button:hover {
    background-color : #58A6FF !important;
    color            : white    !important;
}

/* ── Progress Bar ───────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #238636, #2EA043);
}

/* ── Selectbox & Slider ─────────────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"]    > div > div {
    background-color : #161B22;
    border-radius    : 8px;
}

/* ── Tabs ───────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    color         : #8B949E;
    border-radius : 8px 8px 0 0;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color            : white;
    border-bottom    : 2px solid #58A6FF;
    background-color : #161B22;
}

/* ── Alert / Info / Warning ─────────────────────────────── */
[data-testid="stAlert"] {
    border-radius    : 10px;
    background-color : #1C2128 !important;
    border           : 1px solid #30363D;
}

/* ── Divider ────────────────────────────────────────────── */
.cd-divider {
    display     : flex;
    align-items : center;
    gap         : 8px;
    color       : #8B949E;
    font-size   : 0.8rem;
    margin      : 10px 0;
}
.cd-divider::before,
.cd-divider::after {
    content    : '';
    flex       : 1;
    border-top : 1px solid #30363D;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar          { width: 6px; height: 6px; }
::-webkit-scrollbar-track    { background: #0E1117; }
::-webkit-scrollbar-thumb    { background: #30363D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58A6FF; }

/* ── Image border ───────────────────────────────────────── */
[data-testid="stImage"] > img {
    border-radius : 10px;
    border        : 1px solid #30363D;
}

/* ── Download button area ───────────────────────────────── */
.cd-download-area {
    background    : #161B22;
    border        : 1px solid #30363D;
    border-radius : 12px;
    padding       : 16px 20px;
    margin-top    : 12px;
    text-align    : center;
}

</style>
"""


# ============================================================
# INJECT  — panggil sekali di awal app.py
# ============================================================
def inject_styles() -> None:
    """Inject semua custom CSS ke halaman Streamlit."""
    st.markdown(_MAIN_CSS, unsafe_allow_html=True)


# ============================================================
# HELPER — wrapper HTML komponen umum
# (supaya app.py cukup memanggil fungsi, bukan nulis HTML mentah)
# ============================================================

def card_open(title: str = "", icon: str = "") -> str:
    """Buka div .cd-card dengan judul opsional."""
    header = f"<h3>{icon} {title}</h3>" if title else ""
    return f'<div class="cd-card">{header}'


def card_close() -> str:
    return '</div>'


def section_header(label: str) -> None:
    """Render section header bertema gelap."""
    st.markdown(
        f'<div class="cd-section-header">{label}</div>',
        unsafe_allow_html=True
    )


def divider_label(text: str) -> None:
    """Divider bergaya dengan teks di tengah."""
    st.markdown(
        f'<div class="cd-divider">{text}</div>',
        unsafe_allow_html=True
    )


def metric_card(value: str, label: str) -> None:
    """Render metric card kecil."""
    st.markdown(f"""
    <div class="cd-metric">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def download_area(content) -> None:
    """Wrapper area tombol download."""
    st.markdown('<div class="cd-download-area">', unsafe_allow_html=True)
    content()   # callback yang berisi st.download_button(...)
    st.markdown('</div>', unsafe_allow_html=True)
# ============================================================
# CUSTOM CSS
# ============================================================

# import streamlit as st

# st.markdown("""
# <style>

# /* Hide Streamlit Menu */
# #MainMenu {visibility:hidden;}
# footer {visibility:hidden;}
# header {visibility:hidden;}

# /* Font */
# html, body, [class*="css"] {
#     font-family: 'Segoe UI', sans-serif;
# }

# /* Main Background */
# .stApp {
#     background-color: #0E1117;
#     color: white;
# }

# /* Sidebar */
# [data-testid="stSidebar"] {
#     background-color: #161B22;
# }

# /* Card */
# .custom-card {
#     background-color: #161B22;
#     padding: 20px;
#     border-radius: 15px;
#     border: 1px solid #30363D;
#     margin-bottom: 20px;
# }

# /* Result Box */
# .result-box {
#     padding: 20px;
#     border-radius: 12px;
#     margin-top: 15px;
#     border: 1px solid #30363D;
# }

# .result-normal {
#     background-color: rgba(46, 160, 67, 0.15);
#     border-left: 4px solid #2EA043 !important;
# }

# .result-cardio {
#     background-color: rgba(248, 81, 73, 0.15);
#     border-left: 4px solid #F85149 !important;
# }

# /* Info table inside result box */
# .result-box table {
#     width: 100%;
#     margin-top: 10px;
#     font-size: 0.9rem;
#     border-collapse: collapse;
# }

# .result-box table tr td {
#     padding: 5px 4px;
#     vertical-align: top;
# }

# .result-box table tr td:first-child {
#     color: #8B949E;
#     width: 42%;
# }

# /* Stats table inside custom-card */
# .custom-card table {
#     width: 100%;
#     border-collapse: collapse;
#     font-size: 0.9rem;
# }

# .custom-card table tr td {
#     padding: 6px 4px;
#     border-bottom: 1px solid #21262D;
# }

# .custom-card table tr:last-child td {
#     border-bottom: none;
# }

# .custom-card table tr td:first-child {
#     color: #8B949E;
#     width: 55%;
# }

# /* Metric */
# .metric-card {
#     background: #161B22;
#     border: 1px solid #30363D;
#     border-radius: 12px;
#     padding: 15px;
#     text-align: center;
# }

# /* Upload */
# [data-testid="stFileUploader"] {
#     background-color: #161B22;
#     border: 2px dashed #30363D;
#     border-radius: 10px;
#     padding: 10px;
# }

# /* Button */
# .stButton > button {
#     background-color: #238636;
#     color: white;
#     border-radius: 10px;
#     border: none;
# }

# /* Progress */
# .stProgress > div > div > div > div {
#     background-color: #238636;
# }

# /* Section header */
# .section-header {
#     background: linear-gradient(135deg, #161B22, #1C2128);
#     border: 1px solid #30363D;
#     border-radius: 10px;
#     padding: 12px 18px;
#     margin-bottom: 16px;
# }

# /* Divider label */
# .divider-label {
#     display: flex;
#     align-items: center;
#     gap: 8px;
#     color: #8B949E;
#     font-size: 0.85rem;
#     margin: 8px 0;
# }

# </style>
# """, unsafe_allow_html=True)