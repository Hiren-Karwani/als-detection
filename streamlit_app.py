# =============================================================================
# ALS Risk Assessment Platform  –  streamlit_app.py
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import json
import re
import tempfile
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import librosa
import numpy as np
import pandas as pd
import parselmouth
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from parselmouth.praat import call
from scipy import signal
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, recall_score, roc_auc_score,
)

pd.set_option("styler.render.max_elements", 10_000_000)


# =============================================================================
# PAGE CONFIG  (must be the first st.* render call)
# =============================================================================
st.set_page_config(
    page_title="ALS Risk Assessment Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown("""
<style>
.main { background: #f8f9fa; }
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #efefef;
}
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e6e9ef;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,.05);
}
.stButton > button {
    background: #0ea5e9; color: #fff;
    border: none; border-radius: 8px;
    font-weight: 600; transition: background .15s;
}
.stButton > button:hover { background: #0284c7; }
.risk-card {
    border-radius: 14px; padding: 22px 26px;
    max-width: 360px; margin: 12px 0 20px;
}
.risk-card .lbl {
    font-size: .76rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; opacity: .72;
}
.risk-card .score {
    font-size: 3rem; font-weight: 800; line-height: 1.05;
}
.risk-card .cat { font-size: 1rem; font-weight: 600; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
RISK_COLORS = {
    "No Risk":   {"bg": "#e5e7eb", "fg": "#374151"},
    "Low":       {"bg": "#dbeafe", "fg": "#1e40af"},
    "Moderate":  {"bg": "#60a5fa", "fg": "#1e3a8a"},
    "High":      {"bg": "#2563eb", "fg": "#ffffff"},
    "Very High": {"bg": "#1e3a8a", "fg": "#ffffff"},
}
RISK_BINS   = [0, 25, 50, 75, 100]
RISK_LABELS = ["Low", "Moderate", "High", "Very High"]

# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """Load XGBoost model, scaler, feature list, and optional training metrics."""
    model_dir = Path("models")
    try:
        model         = joblib.load(model_dir / "xgb_als_model.pkl")
        scaler        = joblib.load(model_dir / "scaler.pkl")
        feature_names = joblib.load(model_dir / "feature_names.pkl")
        metrics_path  = model_dir / "metrics.json"
        static_metrics = (
            json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        )
        return model, scaler, feature_names, static_metrics
    except FileNotFoundError as exc:
        st.error(f"❌ Model file not found: {exc}")
        return None, None, None, None


# =============================================================================
# AUDIO FEATURE EXTRACTION
# =============================================================================
def extract_audio_features(audio_path: str, target_sr: int = 8000) -> dict | None:
    """Extract spectral + temporal voice biomarkers using librosa."""
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        f: dict = {}

        # Amplitude
        f["Mean_Amplitude"] = float(np.mean(np.abs(y)))
        f["Std_Amplitude"]  = float(np.std(np.abs(y)))
        f["Max_Amplitude"]  = float(np.max(np.abs(y)))

        # Energy
        f["Energy"]         = float(np.sum(y ** 2))
        f["Energy_Entropy"] = float(-np.sum((y**2) * np.log(y**2 + 1e-10)) / len(y))

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        f["ZCR_Mean"] = float(np.mean(zcr))
        f["ZCR_Std"]  = float(np.std(zcr))

        # Spectral centroid
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        f["Spectral_Centroid_Mean"] = float(np.mean(sc))
        f["Spectral_Centroid_Std"]  = float(np.std(sc))

        # Spectral rolloff
        sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        f["Spectral_Rolloff_Mean"] = float(np.mean(sr_feat))
        f["Spectral_Rolloff_Std"]  = float(np.std(sr_feat))

        # MFCCs (13 coefficients × mean & std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            f[f"MFCC_{i}_Mean"] = float(np.mean(mfcc[i]))
            f[f"MFCC_{i}_Std"]  = float(np.std(mfcc[i]))

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            f[f"Chroma_{i}_Mean"] = float(np.mean(chroma[i]))

        # Onset strength
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        f["Onset_Strength_Mean"] = float(np.mean(onset))
        f["Onset_Strength_Std"]  = float(np.std(onset))

        # Fundamental frequency (autocorrelation)
        frame_len = int(sr * 0.025)
        hop_len   = int(sr * 0.010)
        frames    = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
        pitches = []
        for frm in frames.T:
            win = frm * signal.windows.hann(len(frm))
            ac  = np.correlate(win, win, mode="full")[len(win) - 1:]
            if np.max(ac) > 0:
                lags = np.arange(int(sr / 400), int(sr / 50))
                if len(lags) and lags[-1] < len(ac):
                    idx = lags[np.argmax(ac[lags])]
                    if idx > 0:
                        pitches.append(sr / idx)
        if pitches:
            f["F0_Mean"] = float(np.mean(pitches))
            f["F0_Std"]  = float(np.std(pitches))
            f["F0_Min"]  = float(np.min(pitches))
            f["F0_Max"]  = float(np.max(pitches))

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        f["RMS_Mean"] = float(np.mean(rms))
        f["RMS_Std"]  = float(np.std(rms))

        return f

    except Exception as exc:
        st.error(f"Feature extraction error: {exc}")
        return None


def compute_parselmouth_features(audio_path: str) -> dict:
    """Compute clinical voice biomarkers (F0, jitter, shimmer, HNR) via Praat/parselmouth."""
    zeros = dict(
        f0_mean=0.0, f0_std=0.0,
        jitter_local=0.0, jitter_rap=0.0, jitter_ppq5=0.0,
        shimmer_local=0.0, shimmer_apq3=0.0, shimmer_apq5=0.0,
        hnr_mean=0.0,
    )
    feats = dict(zeros)
    try:
        snd = parselmouth.Sound(str(audio_path))

        # F0 / pitch
        try:
            pitch     = call(snd, "To Pitch", 0.0, 75, 500)
            f0_vals   = pitch.selected_array["frequency"]
            nz        = f0_vals[f0_vals > 0]
            feats["f0_mean"] = float(nz.mean()) if len(nz) else 0.0
            feats["f0_std"]  = float(nz.std())  if len(nz) else 0.0
        except Exception:
            pass

        # Jitter & shimmer via PointProcess
        try:
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            for key, cmd, args in [
                ("jitter_local", "Get jitter (local)",  (0, 0, 0.0001, 0.02, 1.3)),
                ("jitter_rap",   "Get jitter (rap)",    (0, 0, 0.0001, 0.02, 1.3)),
                ("jitter_ppq5",  "Get jitter (ppq5)",   (0, 0, 0.0001, 0.02, 1.3)),
            ]:
                try:
                    feats[key] = float(call(pp, cmd, *args))
                except Exception:
                    pass
            for key, cmd, args in [
                ("shimmer_local", "Get shimmer (local)", (0, 0, 0.0001, 0.02, 1.3, 1.6)),
                ("shimmer_apq3",  "Get shimmer (apq3)",  (0, 0, 0.0001, 0.02, 1.3, 1.6)),
                ("shimmer_apq5",  "Get shimmer (apq5)",  (0, 0, 0.0001, 0.02, 1.3, 1.6)),
            ]:
                try:
                    feats[key] = float(call([snd, pp], cmd, *args))
                except Exception:
                    pass
        except Exception:
            pass

        # HNR
        try:
            harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            feats["hnr_mean"] = float(call(harm, "Get mean", 0, 0))
        except Exception:
            pass

    except Exception:
        return zeros

    return feats


# =============================================================================
# FEATURE MAPPING
# =============================================================================
def _detect_phonation(name: str) -> str | None:
    """Infer phonation vowel from a filename."""
    if not isinstance(name, str):
        return None
    low = name.lower()
    stem = low.rsplit(".", 1)[0] if "." in low else low
    for pattern in [r"phonation[_-]?([aeiou])", r"[_-]([aeiou])\.", r"([aeiou])$"]:
        m = re.search(pattern, stem)
        if m:
            return m.group(1)
    return None


def map_extracted_to_model_features(
    extracted: dict,
    feature_names: list,
    phonation: str | None = None,
) -> dict:
    """Map a raw feature dict into the model's expected feature vector (zero-padded)."""
    mapped = {fn: 0.0 for fn in feature_names}
    if not extracted:
        return mapped

    kl = {k.lower(): k for k in extracted}  # lowercase alias → original key

    for fn in feature_names:
        fn_l = fn.lower()

        # Phonation filtering
        if phonation:
            if ("_a" in fn_l or fn_l.endswith("a")) and phonation != "a":
                continue
            if ("_i" in fn_l or fn_l.endswith("i")) and phonation != "i":
                continue

        # MFCC
        if "mfcc" in fn_l:
            m = re.search(r"mfcc[_\- ]?(\d+)", fn_l)
            if m:
                idx = int(m.group(1))
                for suf in ("mean", "std"):
                    k = f"mfcc_{idx}_{suf}"
                    if k in kl:
                        mapped[fn] = float(extracted[kl[k]]); break
            continue

        # Chroma
        if "chroma" in fn_l:
            m = re.search(r"chroma[_\- ]?(\d+)", fn_l)
            if m:
                k = f"chroma_{int(m.group(1))}_mean"
                if k in kl:
                    mapped[fn] = float(extracted[kl[k]])
            continue

        # Spectral / temporal simple rules
        for keywords, src in [
            (("centroid",  "spectral"), "spectral_centroid_mean"),
            (("rolloff",),             "spectral_rolloff_mean"),
            (("rms",),                 "rms_mean"),
            (("zcr",),                 "zcr_mean"),
            (("zero", "crossing"),     "zcr_mean"),
            (("onset",),               "onset_strength_mean"),
            (("energy",),              "energy"),
            (("f0",),                  "f0_mean"),
            (("fundamental",),         "f0_mean"),
            (("pitch",),               "f0_mean"),
        ]:
            if all(kw in fn_l for kw in keywords):
                if src in kl:
                    mapped[fn] = float(extracted[kl[src]])
                break

        # Clinical jitter J1/J3/J5
        mj = re.match(r"^j(\d+)", fn_l)
        if mj:
            jlookup = {"1": "jitter_local", "3": "jitter_rap", "03": "jitter_rap",
                       "5": "jitter_ppq5", "55": "jitter_ppq5"}
            k = jlookup.get(mj.group(1))
            if k and k in kl:
                mapped[fn] = float(extracted[kl[k]]); continue

        # Clinical shimmer S1/S3/S5
        ms = re.match(r"^s(\d+)", fn_l)
        if ms:
            slookup = {"1": "shimmer_local", "3": "shimmer_apq3", "03": "shimmer_apq3",
                       "5": "shimmer_apq5", "05": "shimmer_apq5"}
            k = slookup.get(ms.group(1))
            if k and k in kl:
                mapped[fn] = float(extracted[kl[k]]); continue

        # Cepstral CCa/CCi
        mcc = re.match(r"^cc([ai])\(?([0-9]+)\)?", fn_l)
        if mcc:
            k = f"mfcc_{max(0, int(mcc.group(2)) - 1)}_mean"
            if k in kl:
                mapped[fn] = float(extracted[kl[k]]); continue

        # Exact key fallback
        if fn in extracted:
            try:
                mapped[fn] = float(extracted[fn])
            except (TypeError, ValueError):
                pass

    return mapped


# =============================================================================
# ANALYSIS PIPELINE  (single DRY function used by both live and batch paths)
# =============================================================================
def run_audio_analysis(audio_path: str, model, scaler, feature_names: list) -> dict:
    """
    Extract features → map → scale → predict.
    Returns: {risk, category, features_raw, error}
    """
    result = {"risk": None, "category": None, "features_raw": {}, "error": None}
    try:
        raw = extract_audio_features(audio_path) or {}
        raw.update(compute_parselmouth_features(audio_path))
        result["features_raw"] = raw

        phon   = _detect_phonation(Path(audio_path).name)
        mapped = map_extracted_to_model_features(raw, feature_names, phonation=phon)

        X_sc  = scaler.transform(pd.DataFrame([mapped], columns=feature_names).values)
        proba = float(model.predict_proba(X_sc)[:, 1][0])
        risk  = round(proba * 100, 2)

        cat = pd.cut([risk], bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True)[0]
        result["risk"]     = risk
        result["category"] = "No Risk" if risk == 0 else str(cat)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def apply_risk_categories(df: pd.DataFrame, col: str = "ALS_Risk_Probability") -> pd.DataFrame:
    """Append Risk_Category to a dataframe that already has probability scores."""
    df["Risk_Category"] = pd.cut(
        df[col], bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True
    ).astype(object)
    df.loc[df[col] == 0, "Risk_Category"] = "No Risk"
    df["Risk_Category"] = df["Risk_Category"].fillna("No Risk")
    return df


# =============================================================================
# CHART HELPERS
# =============================================================================
def gauge_chart(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#0ea5e9"},
            "steps": [
                {"range": [0,   25], "color": "#dbeafe"},
                {"range": [25,  50], "color": "#93c5fd"},
                {"range": [50,  75], "color": "#2563eb"},
                {"range": [75, 100], "color": "#1e3a8a"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def risk_card_html(risk: float, category: str) -> str:
    c = RISK_COLORS.get(category, RISK_COLORS["No Risk"])
    return (
        f'<div class="risk-card" style="background:{c["bg"]};color:{c["fg"]}">'
        f'<div class="lbl">ALS Risk Assessment</div>'
        f'<div class="score">{risk:.1f}%</div>'
        f'<div class="cat">{category}</div>'
        f'</div>'
    )


# =============================================================================
# BROWSER RECORDER COMPONENT
# =============================================================================


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🧬 ALS Platform")
    st.markdown("---")

    model, scaler, feature_names, static_metrics = load_artifacts()

    st.markdown("### Input Mode")
    input_mode = st.radio(
        "Select input mode",
        ["📊 Tabular Data", "🎙️ Voice Samples", "🎤 Live Recording"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    uploaded_file  = None
    uploaded_files = None
    voice_mode     = False

    if input_mode == "📊 Tabular Data":
        st.markdown("**Upload patient CSV**")
        uploaded_file = st.file_uploader(
            "CSV with biomarker columns", type=["csv"],
            label_visibility="collapsed",
        )

    elif input_mode == "🎙️ Voice Samples":
        voice_mode = True
        st.markdown("**Upload audio files**")
        uploaded_files = st.file_uploader(
            "WAV / MP3 / FLAC / OGG",
            type=["wav", "mp3", "flac", "ogg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) ready")

    elif input_mode == "🎤 Live Recording":
        voice_mode = True
        st.markdown("**Live voice recorder**")
        st.info("Use the recorder in the main area. Analysis runs instantly after you stop — no upload needed.")

    st.markdown("---")
    if model is not None and feature_names:
        st.markdown("**Model info**")
        st.caption(f"Input features: {len(feature_names)}")
        if static_metrics:
            for k in ("accuracy", "roc_auc", "recall"):
                v = static_metrics.get(k)
                if v is not None:
                    st.caption(f"{k.replace('_',' ').title()}: {v}")
    elif model is None:
        st.error("Model files missing — place them in `models/`")


# =============================================================================
# MAIN AREA — HEADER
# =============================================================================
st.title("🧬 ALS Risk Assessment Platform")
st.markdown(
    "AI-powered early detection and risk stratification of "
    "Amyotrophic Lateral Sclerosis from voice biomarkers and clinical data."
)
st.divider()

if model is None:
    st.stop()

# =============================================================================
# ── LIVE RECORDING ────────────────────────────────────────────────────────────
# =============================================================================
if input_mode == "🎤 Live Recording":
    # st.audio_input is available in Streamlit >= 1.31.
    # It captures audio entirely inside Streamlit (no separate server, no CORS, no page reload).
    HAS_AUDIO_INPUT = hasattr(st, "audio_input")

    col_rec, col_result = st.columns([1, 1], gap="large")

    with col_rec:
        st.subheader("🎙️ Record Your Voice")

        if HAS_AUDIO_INPUT:
            audio_data = st.audio_input(
                "Click the microphone to start / stop recording",
                key="live_mic",
                label_visibility="visible",
            )
        else:
            # Fallback for Streamlit < 1.31 — manual file upload
            st.info(
                "Your Streamlit version does not support the built-in microphone widget. "
                "Please record a voice memo on your device and upload it below."
            )
            audio_data = st.file_uploader(
                "Upload voice recording (WAV / MP3 / FLAC / OGG)",
                type=["wav", "mp3", "flac", "ogg"],
                key="live_file_upload",
                label_visibility="collapsed",
            )

        st.markdown("---")
        st.caption(
            "💡 **Tips:** Sustain a vowel ('aaah') for 5-15 s · "
            "Stay 15-20 cm from the mic · Quiet room"
        )

        if audio_data is not None:
            if st.button("🔄 Record New Sample", key="btn_retake"):
                st.session_state.pop("live_result", None)
                st.rerun()

    with col_result:
        st.subheader("📊 Analysis Result")

        if audio_data is None:
            st.info("Record or upload audio on the left — results appear here instantly.")
        else:
            # Cache result in session_state so it survives widget state changes
            # Use the audio bytes as a cache key to avoid re-running on unchanged input
            audio_bytes = audio_data.read() if hasattr(audio_data, "read") else bytes(audio_data)

            cached = st.session_state.get("live_result")
            if cached is None or cached.get("_key") != hash(audio_bytes):
                # Write to temp file and run analysis
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                st.audio(tmp_path)

                with st.spinner("🔬 Extracting features and running analysis…"):
                    res = run_audio_analysis(tmp_path, model, scaler, feature_names)

                res["_key"]     = hash(audio_bytes)
                res["tmp_path"] = tmp_path
                st.session_state["live_result"] = res
            else:
                res = cached
                try:
                    st.audio(res["tmp_path"])
                except Exception:
                    pass

            if res.get("error"):
                st.error(f"Analysis error: {res['error']}")
                st.info("Try recording again with less background noise, or upload a cleaner audio file.")
            else:
                st.markdown(risk_card_html(res["risk"], res["category"]),
                            unsafe_allow_html=True)

                out_df = pd.DataFrame([{
                    "Sample":       "live_recording.wav",
                    "ALS Risk (%)": res["risk"],
                    "Category":     res["category"],
                }])
                st.dataframe(out_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Result (CSV)",
                    out_df.to_csv(index=False).encode(),
                    "live_analysis.csv", "text/csv",
                )

                with st.expander("🔍 Extracted features"):
                    fd = pd.DataFrame(
                        res["features_raw"].items(), columns=["Feature", "Value"]
                    ).sort_values("Feature")
                    st.dataframe(fd, use_container_width=True, hide_index=True, height=260)

    st.stop()

# =============================================================================
# ── VOICE SAMPLES ─────────────────────────────────────────────────────────────
# =============================================================================
if input_mode == "🎙️ Voice Samples":
    if not uploaded_files:
        st.info("👈 Upload audio files from the sidebar to begin.")
        st.stop()

    # ── CSS for the scrollable file panel ─────────────────────────────────────
    st.markdown("""
    <style>
    .vs-panel {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }
    .vs-panel-header {
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 12px 16px;
        font-weight: 700;
        font-size: .88rem;
        color: #374151;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .vs-scroll {
        max-height: 340px;
        overflow-y: auto;
        padding: 6px 0;
    }
    .vs-scroll::-webkit-scrollbar { width: 5px; }
    .vs-scroll::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
    .vs-file-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 16px;
        border-bottom: 1px solid #f3f4f6;
        font-size: .82rem;
        color: #374151;
        transition: background .12s;
    }
    .vs-file-row:last-child { border-bottom: none; }
    .vs-file-row:hover { background: #f0f9ff; }
    .vs-file-row .icon { font-size: .9rem; flex-shrink: 0; }
    .vs-file-row .name {
        flex: 1; overflow: hidden;
        text-overflow: ellipsis; white-space: nowrap;
    }
    .vs-badge {
        font-size: .7rem; font-weight: 600;
        padding: 2px 8px; border-radius: 999px;
        white-space: nowrap; flex-shrink: 0;
    }
    .vs-badge.processing { background: #fef3c7; color: #92400e; }
    .vs-badge.done       { background: #d1fae5; color: #065f46; }
    .vs-badge.error      { background: #fee2e2; color: #991b1b; }
    </style>
    """, unsafe_allow_html=True)

    # ── Two-column layout: file panel (left) + results (right) ───────────────
    col_files, col_results = st.columns([1, 2], gap="large")

    with col_files:
        # Build HTML file list panel
        n = len(uploaded_files)
        rows_html = ""
        for up in uploaded_files:
            rows_html += (
                f'<div class="vs-file-row">'
                f'<span class="icon">🎵</span>'
                f'<span class="name" title="{up.name}">{up.name}</span>'
                f'<span class="vs-badge processing">queued</span>'
                f'</div>'
            )

        st.markdown(
            f'''<div class="vs-panel">
              <div class="vs-panel-header">🎙️ Files queued &nbsp;
                <span style="background:#dbeafe;color:#1e40af;font-size:.72rem;
                             padding:2px 8px;border-radius:999px;font-weight:600">
                  {n}
                </span>
              </div>
              <div class="vs-scroll">{rows_html}</div>
            </div>''',
            unsafe_allow_html=True,
        )

    # ── Feature extraction (runs in background while user sees the panel) ─────
    with col_results:
        st.subheader("Processing")
        progress_bar = st.progress(0, text="Starting extraction…")
        status_area  = st.empty()

    features_list: list[dict] = []
    errors: list[str] = []

    for i, up in enumerate(uploaded_files):
        pct  = int((i / n) * 100)
        progress_bar.progress(pct, text=f"Extracting: {up.name}")
        status_area.caption(f"File {i+1} of {n}: {up.name}")

        suffix = Path(up.name).suffix or ".wav"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(up.getbuffer())
                tmp_path = tmp.name

            raw = extract_audio_features(tmp_path) or {}
            raw.update(compute_parselmouth_features(tmp_path))

            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

            raw["Sample_Name"] = up.name
            features_list.append(raw)
        except Exception as exc:
            errors.append(f"{up.name}: {exc}")

    progress_bar.progress(100, text="Extraction complete")
    status_area.empty()

    # Update file panel with done/error badges
    rows_html = ""
    done_names  = {r["Sample_Name"] for r in features_list}
    error_names = {e.split(":")[0] for e in errors}
    for up in uploaded_files:
        if up.name in done_names:
            badge = '<span class="vs-badge done">done</span>'
        elif up.name in error_names:
            badge = '<span class="vs-badge error">error</span>'
        else:
            badge = '<span class="vs-badge processing">skipped</span>'
        rows_html += (
            f'<div class="vs-file-row">'
            f'<span class="icon">🎵</span>'
            f'<span class="name" title="{up.name}">{up.name}</span>'
            f'{badge}'
            f'</div>'
        )

    with col_files:
        st.markdown(
            f'''<div class="vs-panel" style="margin-top:12px">
              <div class="vs-panel-header">✅ Extraction complete &nbsp;
                <span style="background:#d1fae5;color:#065f46;font-size:.72rem;
                             padding:2px 8px;border-radius:999px;font-weight:600">
                  {len(features_list)}/{n}
                </span>
              </div>
              <div class="vs-scroll">{rows_html}</div>
            </div>''',
            unsafe_allow_html=True,
        )
        if errors:
            with st.expander(f"⚠️ {len(errors)} file(s) failed"):
                for e in errors:
                    st.caption(e)

    if not features_list:
        with col_results:
            st.error("No features could be extracted from any uploaded file.")
        st.stop()

    feat_df = pd.DataFrame(features_list)

    # Map → scale → predict
    mapped_rows = [
        map_extracted_to_model_features(
            row.to_dict(), feature_names,
            phonation=_detect_phonation(row.get("Sample_Name", "")),
        )
        for _, row in feat_df.iterrows()
    ]

    with col_results:
        try:
            X_sc   = scaler.transform(pd.DataFrame(mapped_rows, columns=feature_names).values)
            probas = model.predict_proba(X_sc)[:, 1]

            feat_df["ALS_Risk_Probability"] = (probas * 100).round(2)
            feat_df = apply_risk_categories(feat_df)

            summ = feat_df[["Sample_Name", "ALS_Risk_Probability", "Risk_Category"]].copy()
            summ.columns = ["Sample", "ALS Risk (%)", "Risk Category"]

            # KPI row
            kp1, kp2, kp3 = st.columns(3)
            kp1.metric("Samples analysed", len(summ))
            kp2.metric("Avg. Risk",  f"{summ['ALS Risk (%)'].mean():.1f}%")
            kp3.metric("High Risk (>75%)", int((summ['ALS Risk (%)'] > 75).sum()))

            st.subheader("Prediction Results")

            # Colour-coded scrollable results table via dataframe
            st.dataframe(
                summ.style.background_gradient(
                    subset=["ALS Risk (%)"], cmap="Blues", vmin=0, vmax=100
                ).format({"ALS Risk (%)": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
                height=min(38 * len(summ) + 40, 320),   # compact, max 320px
            )

            # Charts side-by-side
            ch1, ch2 = st.columns(2)
            with ch1:
                fig_bar = px.bar(
                    summ.sort_values("ALS Risk (%)"),
                    x="ALS Risk (%)", y="Sample", orientation="h",
                    color="ALS Risk (%)", color_continuous_scale="Blues",
                    title="Risk per Sample",
                )
                fig_bar.update_layout(
                    template="plotly_white",
                    margin=dict(t=40, b=10, l=10, r=10),
                    height=max(200, 28 * len(summ)),
                    yaxis_title=None,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with ch2:
                fig_pie = px.pie(
                    summ, names="Risk Category",
                    color="Risk Category",
                    color_discrete_map={k: v["bg"] for k, v in RISK_COLORS.items()},
                    title="Risk Distribution",
                    hole=0.45,
                )
                fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=260)
                st.plotly_chart(fig_pie, use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "📥 Download Results (CSV)",
                    summ.to_csv(index=False).encode(),
                    "voice_predictions.csv", "text/csv",
                )
            with dl2:
                st.download_button(
                    "📥 Download Full Features (CSV)",
                    feat_df.to_csv(index=False).encode(),
                    "voice_analysis_full.csv", "text/csv",
                    key="dl-full-feats",
                )

            with st.expander("🔍 Raw extracted features table"):
                st.dataframe(feat_df, use_container_width=True, height=280)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.exception(exc)

    # ── Optional labels for performance metrics ────────────────────────────────
    st.markdown("---")
    with st.expander("📐 Upload Labels to Evaluate Model Performance", expanded=False):
        st.caption("CSV must have columns: `Sample_Name` and `label` (0 = No ALS, 1 = ALS)")
        lbl_file = st.file_uploader(
            "Labels CSV", type=["csv"], key="lbl_upload", label_visibility="collapsed"
        )
        if lbl_file:
            try:
                lbl_df = pd.read_csv(lbl_file)
                if {"Sample_Name", "label"}.issubset(lbl_df.columns):
                    merged = feat_df.merge(lbl_df[["Sample_Name", "label"]],
                                           on="Sample_Name", how="left")
                    yt     = merged["label"].fillna(0).astype(int)
                    yp     = (merged["ALS_Risk_Probability"] / 100 >= 0.5).astype(int)
                    yproba = merged["ALS_Risk_Probability"] / 100

                    gm1, gm2, gm3 = st.columns(3)
                    gm1.plotly_chart(gauge_chart(accuracy_score(yt, yp),   "Accuracy"),
                                     use_container_width=True)
                    try:
                        gm2.plotly_chart(gauge_chart(roc_auc_score(yt, yproba), "ROC AUC"),
                                         use_container_width=True)
                    except Exception:
                        gm2.metric("ROC AUC", "N/A (single class)")
                    gm3.plotly_chart(
                        gauge_chart(recall_score(yt, yp, zero_division=0), "Recall"),
                        use_container_width=True,
                    )

                    c_cm, c_rep = st.columns([1, 2])
                    with c_cm:
                        cm = confusion_matrix(yt, yp)
                        st.plotly_chart(
                            px.imshow(cm, text_auto=True,
                                      x=["No ALS", "ALS"], y=["No ALS", "ALS"],
                                      labels=dict(x="Predicted", y="Actual"),
                                      color_continuous_scale="Blues",
                                      title="Confusion Matrix"),
                            use_container_width=True,
                        )
                    with c_rep:
                        st.markdown("#### Classification Report")
                        rep = classification_report(yt, yp, output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(rep).T.style.format("{:.3f}"),
                                     use_container_width=True)
                else:
                    st.error("Labels CSV must have `Sample_Name` and `label` columns.")
            except Exception as exc:
                st.error(f"Error reading labels: {exc}")

    st.stop()
# =============================================================================
# ── TABULAR DATA ──────────────────────────────────────────────────────────────
# =============================================================================
if not uploaded_file:
    # Welcome screen
    st.markdown("### 👋 Welcome")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "**📊 Tabular Data**  \n"
            "Upload a CSV with patient biomarker columns. "
            "The model predicts ALS risk for each row."
        )
    with c2:
        st.markdown(
            "**🎙️ Voice Samples**  \n"
            "Upload WAV / MP3 / FLAC / OGG audio files. "
            "Voice biomarkers are extracted and scored automatically."
        )
    with c3:
        st.markdown(
            "**🎤 Live Recording**  \n"
            "Use your microphone directly in the browser. "
            "Analysis runs instantly after you stop recording."
        )
    st.stop()

# ── Load CSV ──────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

# ── Auto-detect label column ──────────────────────────────────────────────────
LABEL_CANDIDATES = ["label", "Diagnosis(ALS)", "Diagnosis (ALS)", "Diagnosis", "ALS"]
label_col = None
for c in LABEL_CANDIDATES:
    if c in df.columns:
        label_col = c; break
if label_col is None:
    norms = {c.lower().replace(" ", "") for c in LABEL_CANDIDATES}
    for c in df.columns:
        if c.lower().replace(" ", "") in norms:
            label_col = c; break

has_labels = label_col is not None
y_true     = None
if has_labels:
    try:
        y_true = df[label_col].astype(int)
    except Exception:
        st.warning(f"Label column `{label_col}` could not be parsed as 0/1. Ignoring labels.")
        has_labels = False

# ── Feature matrix ────────────────────────────────────────────────────────────
X = pd.DataFrame(0, index=df.index, columns=feature_names)
present = [c for c in feature_names if c in df.columns]
missing = [c for c in feature_names if c not in df.columns]
if missing:
    st.warning(f"⚠️ {len(missing)} feature(s) not in CSV — imputed with 0.")
X[present] = df[present]

# ── Predict ───────────────────────────────────────────────────────────────────
try:
    X_sc   = scaler.transform(X.values)
    probas = model.predict_proba(X_sc)[:, 1]
except Exception as exc:
    st.error(f"Prediction error: {exc}")
    st.stop()

df["ALS_Risk_Probability"] = (probas * 100).round(2)
df = apply_risk_categories(df)

age_col = next((c for c in df.columns if c.lower() == "age"), None)

# =============================================================================
# KPI BAR
# =============================================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Patients",         f"{len(df):,}")
k2.metric("Avg. Risk Score",        f"{df['ALS_Risk_Probability'].mean():.1f}%")
high_risk = int((df["ALS_Risk_Probability"] > 75).sum())
k3.metric("High-Risk Cases (>75%)", high_risk,
          delta="⚠️ review" if high_risk else None, delta_color="inverse")
if has_labels:
    acc = accuracy_score(y_true, (probas >= 0.5).astype(int))
    k4.metric("Model Accuracy (Batch)", f"{acc * 100:.1f}%")
else:
    k4.metric("Prediction Status", "✅ Active")

st.divider()

# =============================================================================
# TABS
# =============================================================================
tab_dash, tab_patients, tab_model = st.tabs(
    ["📊 Executive Dashboard", "📋 Patient Analysis", "🎯 Model Performance"]
)

# ── TAB 1: EXECUTIVE DASHBOARD ────────────────────────────────────────────────
with tab_dash:
    r1, r2 = st.columns(2)

    with r1:
        fig_pie = px.pie(
            df, names="Risk_Category",
            title="Patient Segmentation by Risk Level",
            color="Risk_Category",
            color_discrete_map={k: v["bg"] for k, v in RISK_COLORS.items()},
            hole=0.4,
        )
        fig_pie.update_layout(template="plotly_white", margin=dict(t=60))
        st.plotly_chart(fig_pie, use_container_width=True)

    with r2:
        fig_hist = px.histogram(
            df, x="ALS_Risk_Probability", nbins=25,
            title="Distribution of Risk Probabilities",
            labels={"ALS_Risk_Probability": "Risk Score (%)"},
            color_discrete_sequence=["#0ea5e9"],
        )
        fig_hist.update_layout(
            bargap=0.08, showlegend=False,
            template="plotly_white", margin=dict(t=60),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    if age_col:
        df_age = df.groupby(age_col)["ALS_Risk_Probability"].mean().reset_index()
        fig_line = px.line(
            df_age, x=age_col, y="ALS_Risk_Probability",
            title="Average Risk Trajectory by Age",
            labels={age_col: "Age", "ALS_Risk_Probability": "Avg Risk (%)"},
            markers=True,
        )
        fig_line.update_traces(line_color="#2563eb", line_width=3)
        fig_line.update_layout(template="plotly_white", margin=dict(t=60))
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("#### Risk Category Summary")
    cat_summ = (
        df.groupby("Risk_Category")["ALS_Risk_Probability"]
        .agg(Count="count", Mean="mean", Max="max")
        .reset_index()
        .rename(columns={
            "Risk_Category": "Category",
            "Mean": "Avg Risk (%)",
            "Max":  "Max Risk (%)",
        })
    )
    cat_summ[["Avg Risk (%)", "Max Risk (%)"]] = \
        cat_summ[["Avg Risk (%)", "Max Risk (%)"]].round(1)
    st.dataframe(cat_summ, use_container_width=True, hide_index=True)


# ── TAB 2: PATIENT ANALYSIS ───────────────────────────────────────────────────
with tab_patients:
    st.subheader("Detailed Patient Stratification")

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        min_r, max_r = st.slider("Risk probability (%)", 0, 100, (0, 100))
    with f2:
        search = st.text_input("Search by any column value",
                               placeholder="e.g. patient ID…")
    with f3:
        sort_by = st.selectbox("Sort by", ["ALS_Risk_Probability", "Risk_Category"])

    mask    = (df["ALS_Risk_Probability"] >= min_r) & (df["ALS_Risk_Probability"] <= max_r)
    df_filt = df[mask].copy()

    if search.strip():
        hits = df_filt.apply(
            lambda row: row.astype(str)
                           .str.contains(search.strip(), case=False).any(),
            axis=1,
        )
        df_filt = df_filt[hits]

    df_filt = df_filt.sort_values(sort_by, ascending=False)
    st.caption(f"Showing {len(df_filt):,} of {len(df):,} patients")

    if df_filt.size > pd.get_option("styler.render.max_elements"):
        st.warning("Table too large for gradient formatting — plain view shown.")
        st.dataframe(df_filt, use_container_width=True, height=520)
    else:
        st.dataframe(
            df_filt.style.background_gradient(
                subset=["ALS_Risk_Probability"], cmap="Blues", vmin=0, vmax=100
            ).format({"ALS_Risk_Probability": "{:.2f}%"}),
            use_container_width=True,
            height=520,
        )

    st.download_button(
        "📥 Download Filtered Results (CSV)",
        df_filt.to_csv(index=False).encode(),
        "als_risk_filtered.csv", "text/csv",
        key="dl-filtered",
    )


# ── TAB 3: MODEL PERFORMANCE ──────────────────────────────────────────────────
with tab_model:
    if has_labels and y_true is not None:
        preds = (probas >= 0.5).astype(int)
        st.subheader("Diagnostic Performance — Current Batch")

        gm1, gm2, gm3 = st.columns(3)
        gm1.plotly_chart(gauge_chart(accuracy_score(y_true, preds), "Accuracy"),
                         use_container_width=True)
        try:
            gm2.plotly_chart(gauge_chart(roc_auc_score(y_true, probas), "ROC AUC"),
                             use_container_width=True)
        except Exception:
            gm2.metric("ROC AUC", "N/A — single class in batch")
        gm3.plotly_chart(
            gauge_chart(recall_score(y_true, preds, zero_division=0), "Recall (Sensitivity)"),
            use_container_width=True,
        )

        c_cm, c_rep = st.columns([1, 2])
        with c_cm:
            cm = confusion_matrix(y_true, preds)
            st.plotly_chart(
                px.imshow(cm, text_auto=True,
                          x=["No ALS", "ALS"], y=["No ALS", "ALS"],
                          labels=dict(x="Predicted", y="Actual"),
                          color_continuous_scale="Blues",
                          title="Confusion Matrix"),
                use_container_width=True,
            )
        with c_rep:
            st.markdown("#### Classification Report")
            rep = classification_report(y_true, preds, output_dict=True, zero_division=0)
            st.dataframe(
                pd.DataFrame(rep).T.style.format("{:.3f}"),
                use_container_width=True,
            )
    else:
        st.info(
            "No ground-truth labels found in the uploaded CSV.  \n"
            "Add a `label` column (0 = No ALS, 1 = ALS) to enable performance metrics."
        )

    if static_metrics:
        st.divider()
        st.markdown("### 📚 Reference Training Metrics")
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Training Accuracy", static_metrics.get("accuracy", "N/A"))
        tc2.metric("Training ROC AUC",  static_metrics.get("roc_auc",  "N/A"))
        tc3.metric("Training Recall",   static_metrics.get("recall",   "N/A"))