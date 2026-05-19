"""
News Classifier AI — Streamlit App
COMSATS University Lahore | BS AI | Spring 2026

Team:
  Muhammad Ahad Bashir   (SP23-BAI-030) — Lead
  Muhammad Huzaifa Ali   (SP24-BAI-034) — NLP
  Moiz Ul Rehman         (SP24-BAI-025) — Data
"""

import sys
import os
import time

import streamlit as st

# ── Path setup (works whether launched from project root or app/) ─────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="News Classifier AI",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Root variables ── */
    :root {
        --ink:        #0f1117;
        --paper:      #fafaf8;
        --muted:      #6b7280;
        --border:     #e5e7eb;
        --real-bg:    #d1fae5;
        --real-text:  #065f46;
        --real-ring:  #10b981;
        --fake-bg:    #fee2e2;
        --fake-text:  #991b1b;
        --fake-ring:  #ef4444;
        --accent:     #1a1a2e;
        --highlight:  #e8b86d;
        --card:       #ffffff;
        --radius:     10px;
    }

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    /* ── Main background ── */
    .stApp {
        background-color: var(--paper);
        background-image: radial-gradient(circle at 80% 10%, #fff9ee 0%, transparent 60%),
                          radial-gradient(circle at 10% 90%, #eef6ff 0%, transparent 55%);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--accent);
        border-right: 1px solid #2d2d4a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.55rem !important;
        color: var(--highlight) !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem !important;
        color: #94a3b8 !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2d2d4a !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        text-align: left !important;
        padding: 0.4rem 0.7rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(232,184,109,0.15) !important;
        border-color: var(--highlight) !important;
        color: var(--highlight) !important;
    }

    /* ── Page title ── */
    .page-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: var(--accent);
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0;
    }
    .page-subtitle {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ── Cards ── */
    .result-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .card-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.4rem;
    }
    .topic-name {
        font-family: 'DM Serif Display', serif;
        font-size: 1.9rem;
        color: var(--accent);
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 0.6rem;
    }

    /* ── Verdict badges ── */
    .verdict-real {
        display: inline-block;
        background: var(--real-bg);
        color: var(--real-text);
        border: 1.5px solid var(--real-ring);
        border-radius: 50px;
        padding: 0.35rem 1.1rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.08em;
    }
    .verdict-fake {
        display: inline-block;
        background: var(--fake-bg);
        color: var(--fake-text);
        border: 1.5px solid var(--fake-ring);
        border-radius: 50px;
        padding: 0.35rem 1.1rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.08em;
    }

    /* ── Confidence label ── */
    .confidence-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: var(--muted);
        margin-top: 0.5rem;
    }

    /* ── Timing ── */
    .timing-pill {
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: var(--muted);
        background: #f3f4f6;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        margin-top: 0.8rem;
        border: 1px solid var(--border);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid var(--border);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--muted);
        padding: 0.6rem 1.2rem;
        border-radius: 0;
        background: transparent !important;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
        font-weight: 600 !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.5rem !important;
        letter-spacing: 0.01em;
        box-shadow: 0 2px 8px rgba(26,26,46,0.2) !important;
        transition: opacity 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.88 !important;
    }

    /* ── Warning / error adjustments ── */
    .stAlert {
        border-radius: var(--radius) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
    }

    /* ── Divider color ── */
    hr { border-color: var(--border) !important; }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.models.pipeline import NewsPipeline  # noqa: E402
    return NewsPipeline(models_dir="saved_models")

pipeline_error: str | None = None
pipeline = None

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    pipeline_error = "Models not trained yet. Run:  **`python app/train_models.py`**"
except ModuleNotFoundError as e:
    pipeline_error = f"Missing module: `{e.name}`. Check your install."
except Exception as e:
    pipeline_error = f"Failed to load pipeline: {e}"

# ── Session state defaults ────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "balloons_fired" not in st.session_state:
    st.session_state["balloons_fired"] = False

# ── Demo examples ─────────────────────────────────────────────────────────────
EXAMPLES = [
    ("⚽  Sports",    "Manchester United defeats Arsenal 2-1 in Premier League clash"),
    ("💻  Tech",      "Apple announces M4 chip with 40% faster neural engine"),
    ("🤖  Satire",    "BREAKING: President secretly replaced by AI robot, sources say"),
    ("🍫  Hoax",      "Scientists confirm chocolate cures all cancers, FDA suppresses truth"),
    ("📈  Finance",   "Stock markets surge as Fed holds interest rates steady"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🗞️ News Classifier AI")
    st.markdown(
        "<div style='font-size:0.72rem; color:#94a3b8; font-family:DM Mono,monospace; "
        "letter-spacing:0.1em; text-transform:uppercase; margin-top:-0.4rem; margin-bottom:1rem;'>"
        "COMSATS University Lahore<br>BS Artificial Intelligence · Spring 2026"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 👥 Team")
    st.markdown(
        """
<div style='font-size:0.8rem; line-height:2;'>
  <span style='color:#e2e8f0; font-weight:500;'>Muhammad Ahad Bashir</span>
  <span style='color:#e8b86d; font-family:DM Mono,monospace; font-size:0.68rem;'> SP23-BAI-030 · Lead</span><br>
  <span style='color:#e2e8f0; font-weight:500;'>Muhammad Huzaifa Ali</span>
  <span style='color:#e8b86d; font-family:DM Mono,monospace; font-size:0.68rem;'> SP24-BAI-034 · NLP</span><br>
  <span style='color:#e2e8f0; font-weight:500;'>Moiz Ul Rehman</span>
  <span style='color:#e8b86d; font-family:DM Mono,monospace; font-size:0.68rem;'> SP24-BAI-025 · Data</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 🧪 Quick Demo Examples")
    st.markdown(
        "<div style='font-size:0.73rem; color:#64748b; margin-bottom:0.6rem;'>"
        "Click to load an example into the classifier."
        "</div>",
        unsafe_allow_html=True,
    )
    for icon_label, text in EXAMPLES:
        if st.button(icon_label, use_container_width=True):
            st.session_state["input_text"] = text
            st.rerun()

    st.divider()

    st.markdown("### ℹ️ About")
    st.markdown(
        "<div style='font-size:0.78rem; color:#94a3b8; line-height:1.6;'>"
        "This system uses an NLP pipeline to classify news articles by topic "
        "and detect potential misinformation using ensemble ML models."
        "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='page-title'>News Classifier <em>AI</em></div>"
    "<div class='page-subtitle'>Topic Detection · Misinformation Analysis</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# Block everything if models are missing
if pipeline_error:
    st.error(f"⚠️ {pipeline_error}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_text, tab_image = st.tabs(["📝  Text Input", "🖼️  Image Upload"])


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render classification results
# ─────────────────────────────────────────────────────────────────────────────
def render_results(result: dict, processing_ms: float) -> None:
    """Render topic + verdict cards side by side."""
    topic      = result.get("topic", "Unknown")
    topic_conf = float(result.get("topic_conf", 0.0))
    label      = result.get("verdict", "Real")          # "Real" or "Fake"
    fake_conf  = float(result.get("fake_conf", 0.0))

    col_topic, col_gap, col_verdict = st.columns([5, 0.4, 5])

    with col_topic:
        st.markdown(
            f"""
            <div class='result-card'>
              <div class='card-label'>🏷️ Detected Topic</div>
              <div class='topic-name'>{topic}</div>
              <div class='confidence-label'>Confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(topic_conf, 0.0), 1.0))
        st.markdown(
            f"<div class='confidence-label'>{topic_conf * 100:.1f}% confidence</div>",
            unsafe_allow_html=True,
        )

    with col_verdict:
        is_fake = label.strip().lower() == "fake"
        if is_fake:
            verdict_html = "<span class='verdict-fake'>🚨 FAKE NEWS</span>"
            bar_color    = "normal"   # st.error colours the container
        else:
            verdict_html = "<span class='verdict-real'>✅ REAL NEWS</span>"
            bar_color    = "normal"

        st.markdown(
            f"""
            <div class='result-card'>
              <div class='card-label'>🔍 Credibility Verdict</div>
              <div style='margin: 0.4rem 0 0.7rem;'>{verdict_html}</div>
              <div class='confidence-label'>Fake probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(fake_conf, 0.0), 1.0))
        st.markdown(
            f"<div class='confidence-label'>{fake_conf * 100:.1f}% fake probability</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='timing-pill'>⚡ Processed in {processing_ms:.0f} ms</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state["balloons_fired"]:
        st.balloons()
        st.session_state["balloons_fired"] = True


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Text Input
# ─────────────────────────────────────────────────────────────────────────────
with tab_text:
    st.markdown("<br>", unsafe_allow_html=True)

    user_text = st.text_area(
        label="Article or Headline",
        value=st.session_state.get("input_text", ""),
        key="input_text",
        height=180,
        placeholder="Paste a news article or headline here…",
        help="Supports full articles or single headlines.",
    )

    analyze_btn = st.button("🔍  Analyze Article", type="primary")

    if analyze_btn:
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            try:
                with st.spinner("Analyzing article…"):
                    t0 = time.perf_counter()
                    result = pipeline.analyze(user_text)
                    ms = (time.perf_counter() - t0) * 1000

                st.markdown("<br>", unsafe_allow_html=True)
                render_results(result, ms)

            except Exception as exc:
                st.error(
                    f"Analysis failed: {exc}\n\n"
                    "Please check your model files or try a different input."
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Image Upload
# ─────────────────────────────────────────────────────────────────────────────
with tab_image:
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a news screenshot",
        type=["png", "jpg", "jpeg"],
        help="Upload a screenshot of a news article. OCR will extract the text.",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔎  Extract Text & Classify", type="primary"):
            extracted_text = ""

            # ── OCR ──
            try:
                with st.spinner("Running OCR…"):
                    from src.vision.ocr_reader import OCRReader  # noqa: E402
                    raw_bytes    = uploaded_file.getvalue()
                    extracted_text = OCRReader().read_from_bytes(raw_bytes)

            except RuntimeError as e:
                st.error(
                    "**Tesseract not found.**\n\n"
                    "Install it with one of:\n"
                    "- **macOS**: `brew install tesseract`\n"
                    "- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`\n"
                    "- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki\n\n"
                    f"Details: `{e}`"
                )
            except Exception as e:
                st.error(f"OCR failed unexpectedly: {e}")

            # ── Show extracted text (editable) ──
            if extracted_text:
                st.markdown("**Extracted Text** *(you can edit before classifying)*")
                corrected_text = st.text_area(
                    label="Extracted text",
                    value=extracted_text,
                    height=160,
                    key="ocr_text",
                    label_visibility="collapsed",
                )

                # ── Auto-classify ──
                try:
                    with st.spinner("Classifying…"):
                        t0     = time.perf_counter()
                        result = pipeline.analyze(corrected_text)
                        ms     = (time.perf_counter() - t0) * 1000

                    st.markdown("<br>", unsafe_allow_html=True)
                    render_results(result, ms)

                except Exception as exc:
                    st.error(
                        f"Classification failed: {exc}\n\n"
                        "Try editing the extracted text above and re-running."
                    )