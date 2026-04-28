import os
from datetime import datetime
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
from google import genai
import pandas as pd
from io import BytesIO
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


# LOAD MODELS

@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model("heart_ecg_cnn1.h5")


cnn_model = load_cnn()
ml_model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

classes = ["Normal", "MI", "History_MI", "Abnormal"]

CLINICAL_FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    "age_group", "chol_level", "thalach_level",
    "age_chol_interaction", "oldpeak_log",
]

# Human-readable input metadata (encodings match common UCI-style heart disease dataset)
CLINICAL_HELP = {
    "age": "Patient age in years.",
    "sex": "0 = female, 1 = male (dataset encoding).",
    "cp": "Chest pain type: 0 = typical angina, 1 = atypical angina, 2 = non-anginal, 3 = asymptomatic.",
    "trestbps": "Resting blood pressure in mm Hg (on admission).",
    "chol": "Serum cholesterol in mg/dL.",
    "fbs": "Fasting blood sugar: 0 if ≤120 mg/dL, 1 if higher.",
    "restecg": "Resting ECG: 0/1/2 (e.g. normal, ST–T abnormality, ventricular hypertrophy).",
    "thalach": "Maximum heart rate achieved (exercise).",
    "exang": "Exercise-induced angina: 0 = no, 1 = yes.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of peak exercise ST segment (0, 1, 2).",
    "ca": "Number of major vessels colored by fluoroscopy (0–4).",
    "thal": "Thalassemia blood disorder encoding (0–3).",
}

CP_LABELS = {
    0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"
}


# GEMINI

def _gemini_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        return None


def _gemini_model_name() -> str:
    env = os.environ.get("GEMINI_MODEL", "").strip()
    if env:
        return env
    try:
        return str(st.secrets["GEMINI_MODEL"]).strip()
    except Exception:
        return "gemini-2.5-flash"


def get_ai_suggestion(input_type, data, result):
    api_key = _gemini_api_key()
    if not api_key:
        return (
            "AI interpretation is unavailable. Add a Gemini API key (environment variable "
            "GEMINI_API_KEY or `.streamlit/secrets.toml`) from "
            "https://aistudio.google.com/app/apikey"
        )

    client = genai.Client(api_key=api_key)
    model_name = _gemini_model_name()

    prompt = f"""
    You are an experienced cardiologist AI assistant. A patient has undergone heart disease analysis.

    INPUT TYPE: {input_type}
    PATIENT / MODEL DATA: {data}
    MODEL PREDICTION: {result}

    Provide a concise, structured interpretation (not a formal diagnosis) with:
    1) Risk level (Low / Medium / High) and brief reason
    2) Short explanation in plain language
    3) 2–3 practical lifestyle or follow-up points
    4) When to see a doctor

    Be calm, clear, and avoid causing undue alarm. This is decision support, not a diagnosis.
    """
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = getattr(response, "text", None)
        if not text:
            return "No text returned from the model. Check GEMINI_MODEL and API quota."
        return text
    except Exception as e:
        return f"AI service error: {e!s}"


# CLINICAL HELPERS

def clinical_risk_percent_and_tier(input_scaled) -> tuple[float, str]:
    """Returns risk score 0–100 (P(class 1)) and tier string."""
    try:
        if hasattr(ml_model, "predict_proba"):
            p = ml_model.predict_proba(input_scaled)[0]
            # positive class = index 1 for binary
            p_disease = float(p[1] if len(p) > 1 else p[0])
        else:
            p_disease = 0.5
    except Exception:
        p_disease = 0.5
    risk_pct = round(100.0 * p_disease, 1)
    if risk_pct < 35:
        return risk_pct, "Low"
    if risk_pct < 65:
        return risk_pct, "Medium"
    return risk_pct, "High"


def contributing_factors(age, sex, trestbps, chol, thalach, exang, oldpeak, cp) -> list[str]:
    """Lightweight, explainable flags — not causal attribution."""
    f: list[str] = []
    if age >= 55:
        f.append("Age in a range where cardiovascular risk is often reassessed in screening.")
    if chol >= 240:
        f.append("Cholesterol value that often warrants follow-up in clinical context.")
    if trestbps >= 140:
        f.append("Resting blood pressure in a range that may need monitoring.")
    if thalach < 120 and age < 60:
        f.append("Relatively low maximum heart rate in context of exercise data.")
    if exang == 1:
        f.append("Exercise angina (yes) is often weighted heavily in such models.")
    if oldpeak >= 2.0:
        f.append("ST depression (oldpeak) is a notable ischemia-related signal in many scores.")
    if cp in (0, 1):
        f.append("Chest pain pattern may influence risk in the trained model.")
    if not f:
        f.append("Continue routine preventive care; discuss results with a clinician if unsure.")
    return f[:5]


# -------------------------------
# IMAGE / CNN
# -------------------------------
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = cv2.resize(img, (240, 200))
    return img


def predict_cnn(img):
    cnn_input = img.reshape(1, 200, 240, 1)
    pred = cnn_model.predict(cnn_input, verbose=0)
    class_id = int(np.argmax(pred))
    return classes[class_id], float(np.max(pred))


def waveform_analysis(img):
    wave = np.mean(img, axis=0)
    wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-8)
    peaks, _ = find_peaks(wave, distance=20)
    return wave, peaks, len(peaks) * 6, float(np.std(wave))


# -------------------------------
# PDF
# -------------------------------
def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def generate_report_pdf(
    title: str,
    mode_label: str,
    input_block: str,
    result_block: str,
    ai_text: str,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#1e293b")
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#0f766e"), fontSize=12)

    story = []
    story.append(Paragraph(f"<b>{_escape(title)}</b>", styles["Title"]))
    story.append(Paragraph(f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", body))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Medical disclaimer: This report is for informational support only. It is not a diagnosis or treatment plan. Always consult a qualified clinician.", body))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Mode", h2))
    story.append(Paragraph(_escape(mode_label), body))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Inputs & context", h2))
    story.append(Paragraph(_escape(input_block)[:15000] or "—", body))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Model output", h2))
    story.append(Paragraph(_escape(result_block)[:8000] or "—", body))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("AI summary (non-clinical; not a diagnosis)", h2))
    story.append(Paragraph(_escape(ai_text)[:12000] or "—", body))
    doc.build(story)
    return buffer.getvalue()


# STYLING

def inject_clinical_style():
    st.markdown(
        """
<style>
    :root { --cl-fg: #0f172a; --cl-fg2: #334155; --cl-muted: #64748b; --cl-border: #e2e8f0; }
    /* Base: force readable dark text (fixes dark theme / white text on light bg) */
    .stApp, section[data-testid="stMain"], [data-testid="stAppViewContainer"] {
        color: var(--cl-fg) !important;
        --text-color: var(--cl-fg);
    }
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #f0f6fa 0%, #f4f7fb 100%) !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stCaption"], .stMetric label, h1, h2, h3, h4, h5, h6 {
        color: var(--cl-fg) !important;
    }
    [data-baseweb="tab"] button, [data-baseweb="tab"] p { color: var(--cl-fg2) !important; }
    [data-baseweb="select"] > div, [data-baseweb="input"] { color: var(--cl-fg) !important; }
    label, [data-testid="stWidgetLabel"], .stSelectbox label, .stSlider label { color: var(--cl-fg2) !important; }
    [data-baseweb="accordion"] { color: var(--cl-fg) !important; }
    .stAlert { color: var(--cl-fg) !important; }
    .stProgress > div { color: var(--cl-fg2); }
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: var(--cl-fg) !important; }
    .cl-header { background: #ffffff; border-bottom: 1px solid var(--cl-border); padding: 0.9rem 0 0.4rem; margin: -1rem -4rem 1rem; padding-left: 2rem; padding-right: 2rem; }
    .cl-title { font-family: system-ui, sans-serif; font-size: 1.55rem; font-weight: 700; color: #0f172a !important; letter-spacing: -0.02em; }
    .cl-sub { font-size: 0.8rem; color: #64748b !important; margin-top: 0.35rem; line-height: 1.4; }
    .risk-badge { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .bd-low { background: #e0f2f1; color: #0f766e !important; }
    .bd-med { background: #fef3c7; color: #b45309 !important; }
    .bd-high { background: #fef2f2; color: #b91c1c !important; }
    div[data-testid="stVerticalBlock"] > div:has(> [data-baseweb="tab-list"]) { gap: 0.5rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(tier: str) -> str:
    cls = "bd-high" if tier == "High" else ("bd-med" if tier == "Medium" else "bd-low")
    return f'<span class="risk-badge {cls}">{tier} risk</span>'


# PAGE

st.set_page_config(
    page_title="Cardio Risk Assistant",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_clinical_style()

st.markdown(
    """
<div class="cl-header">
  <div class="cl-title">Cardio Risk Assistant</div>
  <div class="cl-sub">Not a medical diagnosis — for educational and decision-support use only. Always consult a qualified healthcare professional for clinical decisions.</div>
</div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ----- Session defaults -----
if "clinical_result" not in st.session_state:
    st.session_state["clinical_result"] = None
if "ecg_label" not in st.session_state:
    st.session_state["ecg_label"] = None

# ----- Main tabs -----
t_clinical, t_ecg = st.tabs(["Clinical Risk", "ECG Analysis"])


# CLINICAL

with t_clinical:
    st.caption("10-year–style model scores are not guaranteed. Outputs reflect training data limits and are not a substitute for examination.")

    c_left, c_right = st.columns((1, 1.1), gap="large")

    with c_left:
        st.markdown("**Patient & vitals**")
        age = st.slider("Age (years)", 20, 100, 55, help=CLINICAL_HELP["age"])
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help=CLINICAL_HELP["sex"])
        cp = st.selectbox(
            "Chest pain type", options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} — {CP_LABELS.get(x, '')}", help=CLINICAL_HELP["cp"]
        )
        trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 120, help=CLINICAL_HELP["trestbps"])
        chol = st.slider("Cholesterol (mg/dL)", 100, 600, 200, help=CLINICAL_HELP["chol"])
        fbs = st.selectbox("Fasting blood sugar", options=[0, 1], format_func=lambda x: "≤120 mg/dL" if x == 0 else ">120 mg/dL", help=CLINICAL_HELP["fbs"])
        restecg = st.selectbox("Resting ECG (0/1/2)", options=[0, 1, 2], help=CLINICAL_HELP["restecg"])
        thalach = st.slider("Max. heart rate achieved", 70, 220, 150, help=CLINICAL_HELP["thalach"])
        exang = st.selectbox("Exercise angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=CLINICAL_HELP["exang"])
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1, help=CLINICAL_HELP["oldpeak"])
        slope = st.selectbox("ST slope (0/1/2)", options=[0, 1, 2], help=CLINICAL_HELP["slope"])
        ca = st.selectbox("Vessels (0–4)", options=[0, 1, 2, 3, 4], help=CLINICAL_HELP["ca"])
        thal = st.selectbox("Thal (0–3)", options=[0, 1, 2, 3], help=CLINICAL_HELP["thal"])

        run_clinical = st.button("Run prediction", type="primary", use_container_width=True)

    if run_clinical:
        with st.spinner("Scoring and generating summary…"):
            try:
                age_group = pd.cut([age], bins=[0, 40, 55, 70, 100], labels=[0, 1, 2, 3]).astype(int)[0]
                chol_level = pd.cut([chol], bins=[0, 200, 240, 600], labels=[0, 1, 2]).astype(int)[0]
                thalach_level = pd.cut([thalach], bins=[0, 100, 150, 220], labels=[0, 1, 2]).astype(int)[0]
                age_chol_interaction = age * chol
                oldpeak_log = np.log1p(oldpeak)

                input_df = pd.DataFrame([[
                    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
                    slope, ca, thal, age_group, chol_level, thalach_level,
                    age_chol_interaction, oldpeak_log
                ]], columns=CLINICAL_FEATURE_COLS)

                input_scaled = scaler.transform(input_df)
                pred = int(ml_model.predict(input_scaled)[0])
                result_label = "Likely heart disease" if pred == 1 else "Lower likelihood (no disease class)"
                risk_pct, risk_tier = clinical_risk_percent_and_tier(input_scaled)
                factors = contributing_factors(age, sex, trestbps, chol, thalach, exang, oldpeak, cp)
                input_snapshot = (
                    f"Age {age}, sex {sex}, cp {cp}, BP {trestbps}, chol {chol}, FBS {fbs}, restECG {restecg}, "
                    f"max HR {thalach}, exang {exang}, oldpeak {oldpeak}, slope {slope}, ca {ca}, thal {thal}"
                )
                result_snapshot = f"Binary class: {pred} ({result_label}) | Model risk score: {risk_pct}% | Tier: {risk_tier}"

                suggestion = get_ai_suggestion("Clinical Data", input_df.values.tolist(), result_snapshot)
                st.session_state["clinical_result"] = result_label
                st.session_state["clinical_suggestion"] = suggestion
                st.session_state["clinical_risk_pct"] = risk_pct
                st.session_state["clinical_tier"] = risk_tier
                st.session_state["clinical_factors"] = factors
                st.session_state["clinical_input_snapshot"] = input_snapshot
                st.session_state["clinical_result_snapshot"] = result_snapshot
                st.session_state["clinical_pred"] = pred
                st.session_state["clinical_ts"] = datetime.now().isoformat(timespec="seconds")
            except Exception as e:
                st.error(f"Could not complete prediction. Check inputs. ({e!s})")

    with c_right:
        st.markdown("**Results**")
        if not st.session_state.get("clinical_result"):
            st.info("Enter values to the left, then use **Run prediction** to see risk score, tier, and AI summary.")
        else:
            rp = st.session_state.get("clinical_risk_pct", 0)
            tier = st.session_state.get("clinical_tier", "—")
            st.markdown("### Model output")
            st.metric(
                label="Disease risk score (model)",
                value=f"{rp}%",
                help="From the model’s positive-class probability, not a population-calibrated 10-year risk score.",
            )
            st.progress(min(rp / 100.0, 1.0))
            st.markdown(risk_badge_html(tier), unsafe_allow_html=True)
            st.caption("Tier is derived from the model score (e.g. under 35% low, 35–65% medium, higher high).")

            with st.container(border=True):
                st.markdown("**Classification**")
                st.write(st.session_state["clinical_result"])

            with st.expander("Key factors highlighted for discussion", expanded=True):
                for line in st.session_state.get("clinical_factors", []):
                    st.markdown(f"• {line}")
                st.caption("These are discussion prompts for a clinician, not causal proof.")

            with st.expander("AI brief (Gemini) — not a diagnosis", expanded=True):
                st.markdown(st.session_state.get("clinical_suggestion", ""))
                st.caption("Generative text can be wrong; it does not replace a physician.")

            pdf_in = st.session_state.get("clinical_input_snapshot", "")
            pdf_res = st.session_state.get("clinical_result_snapshot", "")
            pdf_bytes = generate_report_pdf(
                "Cardio Risk — Clinical",
                "Clinical tabular model",
                pdf_in,
                pdf_res,
                st.session_state.get("clinical_suggestion", ""),
            )
            st.download_button(
                label="Download report (PDF)",
                data=pdf_bytes,
                file_name=f"cardio_clinical_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with st.expander("How this clinical view works"):
        st.markdown(
            """
- Values are run through the same preprocessing and scaler as the training notebook, then a trained classifier.
- The **% risk** is the model’s estimated probability of the “disease” class, shown for transparency, not as a population-validated 10-year risk.
- Tiers and bullet points are aids for discussion, **not** proof of cause.
            """
        )


# ECG

with t_ecg:
    st.caption("ECG classifiers are data-dependent; confidence is from the CNN softmax, not clinical certainty.")

    e_left, e_right = st.columns((1, 1.1), gap="large")
    with e_left:
        uploaded = st.file_uploader("ECG image (JPG/PNG)", type=["jpg", "png", "jpeg"], help="Use a clear image of the ECG plot.")
        b_row = st.columns(2)
        with b_row[0]:
            run_ecg = st.button("Analyze ECG", type="primary", use_container_width=True, disabled=uploaded is None)
        with b_row[1]:
            if st.session_state.get("ecg_label") is not None and st.button("New analysis", use_container_width=True, key="ecg_reset"):
                for _k in (
                    "ecg_preview", "ecg_label", "ecg_confidence", "ecg_wave", "ecg_peaks",
                    "ecg_hr", "ecg_var", "ecg_suggestion", "ecg_result_snapshot", "ecg_input_snapshot", "ecg_ts",
                ):
                    st.session_state.pop(_k, None)
                st.rerun()

    with e_right:
        if st.session_state.get("ecg_label") is None:
            st.info("Upload a strip image and run **Analyze ECG** to see class, confidence, waveform, and AI summary.")
        else:
            st.caption("Results are shown below. **New analysis** clears the current run.")

    if run_ecg and uploaded is not None:
        with st.spinner("Running CNN and summarizing…"):
            if hasattr(uploaded, "getvalue"):
                raw = uploaded.getvalue()
            else:
                uploaded.seek(0)
                raw = uploaded.read()
            file_bytes = np.asarray(bytearray(raw), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is None:
                st.error("Could not read the image. Try another file format or re-export the image.")
            else:
                processed = preprocess_image(img)
                label, confidence = predict_cnn(processed)
                wave, peaks, hr, var = waveform_analysis(processed)
                result_line = f"ECG class: {label} (confidence {confidence*100:.1f}%)"
                suggestion = get_ai_suggestion("ECG image", f"Class={label}, conf={confidence:.3f}", result_line)
                st.session_state["ecg_preview"] = raw
                st.session_state["ecg_label"] = label
                st.session_state["ecg_confidence"] = confidence
                st.session_state["ecg_wave"] = wave.tolist()
                st.session_state["ecg_peaks"] = peaks.tolist()
                st.session_state["ecg_hr"] = hr
                st.session_state["ecg_var"] = var
                st.session_state["ecg_suggestion"] = suggestion
                st.session_state["ecg_result_snapshot"] = f"{result_line}; HR est. {hr} bpm; signal var. {var:.4f}"
                st.session_state["ecg_input_snapshot"] = "Image uploaded; CNN + waveform heuristics"
                st.session_state["ecg_ts"] = datetime.now().isoformat(timespec="seconds")

    if st.session_state.get("ecg_label") is not None:
        e_left, e_right = st.columns((1, 1.1), gap="large")
        with e_left:
            st.markdown("**Image**")
            if st.session_state.get("ecg_preview"):
                st.image(BytesIO(st.session_state["ecg_preview"]), use_container_width=True)
        with e_right:
            st.markdown("**CNN prediction**")
            lab = st.session_state["ecg_label"]
            conf = st.session_state["ecg_confidence"]
            is_alert = lab in ("MI", "Abnormal")
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Predicted class", lab)
            mcol2.metric("Confidence", f"{100*conf:.1f}%", help="Softmax from the CNN, not a clinical confidence interval.")
            if is_alert:
                st.error("This result category may warrant clinical correlation — it is **not** a diagnosis.")
            else:
                st.success("Model output in a non-urgent category; still not a medical clearance.")

            st.markdown("**Projected 1D waveform (exploratory)**")
            wave = np.array(st.session_state["ecg_wave"])
            peaks = np.array(st.session_state["ecg_peaks"])
            fig, ax = plt.subplots(figsize=(8, 2.2), dpi=100)
            ax.set_facecolor("#f8fafc")
            fig.patch.set_facecolor("#f8fafc")
            ax.plot(wave, color="#0d9488", linewidth=0.9)
            ax.scatter(peaks, wave[peaks], color="#1e3a5f", s=8, zorder=3)
            ax.set_xticks([])
            ax.set_ylabel("Norm. signal")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(f"Estimated heart rate (heuristic from peaks): **{st.session_state['ecg_hr']} bpm** — informational only.")
            st.caption(f"Signal spread (std.): {st.session_state['ecg_var']:.4f}")

            with st.expander("AI brief (Gemini) — not a diagnosis"):
                st.markdown(st.session_state.get("ecg_suggestion", ""))
                st.caption("Image models can misread noise or lead placement; confirm with standard clinical ECG if needed.")

        pdf_bytes = generate_report_pdf(
            "Cardio Risk — ECG",
            "ECG image CNN + waveform heuristics",
            st.session_state.get("ecg_input_snapshot", ""),
            st.session_state.get("ecg_result_snapshot", ""),
            st.session_state.get("ecg_suggestion", ""),
        )
        st.download_button(
            label="Download report (PDF)",
            data=pdf_bytes,
            file_name=f"cardio_ecg_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with st.expander("How ECG mode works"):
        st.markdown(
            """
- A CNN classifies the **image** (not a raw 12-lead signal database).
- The line plot is a downsampled projection of the image, useful only as a **visual** sanity check, not a beat detector.
- **Confidence** = neural network softmax; it does not equal diagnostic certainty in the EMR sense.
            """
        )

st.markdown("---")
st.caption("© Model outputs depend on training data and validation; use only under appropriate supervision. Streamlit + TensorFlow + scikit-learn + Gemini (optional).")
