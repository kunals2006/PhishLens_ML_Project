"""PhishLens: styled URL review prototype."""
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

from url_features import extract_many

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"

st.set_page_config(page_title="PhishLens | URL Review", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --pl-ink:#182b3a; --pl-muted:#627383; --pl-line:#dfe7eb; --pl-teal:#087f73; --pl-soft:#eaf5f2; --pl-red:#b64f40; --pl-red-soft:#fff0ec; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      .stApp { background:linear-gradient(180deg,#f5f8fa 0%,#f2f6f7 100%); color:var(--pl-ink); }
      [data-testid="stHeader"] { background:transparent; }
      [data-testid="stToolbar"] { visibility:hidden; }
      .block-container { max-width:1120px; padding-top:1.1rem; padding-bottom:3.5rem; }
      [data-testid="stVerticalBlock"] { gap:1rem; }
      .pl-topbar { display:flex; align-items:center; justify-content:space-between; padding:12px 0 17px; border-bottom:1px solid var(--pl-line); margin-bottom:30px; }
      .pl-brand { display:flex; align-items:center; gap:11px; color:var(--pl-ink); font:700 19px 'Manrope',sans-serif; letter-spacing:-.5px; }
      .pl-logo { width:38px;height:38px;display:grid;place-items:center;border-radius:12px;background:#e2f2ee;color:#087f73;font-size:21px; }
      .pl-topnote { color:var(--pl-muted); font-size:13px; }
      .pl-hero { margin:0 0 22px; }
      .pl-eyebrow { color:var(--pl-teal); font-size:11px; font-weight:700; letter-spacing:1.35px; text-transform:uppercase; margin-bottom:8px; }
      .pl-title { color:var(--pl-ink); font:800 clamp(30px,4vw,43px)/1.12 'Manrope',sans-serif; letter-spacing:-1.7px; margin:0 0 10px; }
      .pl-subtitle { color:var(--pl-muted); font-size:15px; line-height:1.65; max-width:720px; margin:0; }
      [data-testid="stForm"] { background:#fff; border:1px solid var(--pl-line); border-radius:17px; padding:20px 22px 17px; box-shadow:0 8px 26px rgba(27,53,70,.045); }
      [data-testid="stTextInputRootElement"] input { height:48px; background:#fbfcfd; border:1px solid #d8e1e6; border-radius:10px; color:var(--pl-ink); font-size:15px; }
      [data-testid="stTextInputRootElement"] input:focus, input:focus-visible { border:1px solid #50a99c !important; outline:none !important; box-shadow:0 0 0 2px rgba(8,127,115,.12) !important; }
      label, label *, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color:#334858 !important; font-weight:600; font-size:13px; }
      input { color:#182b3a !important; border-color:#d8e1e6 !important; }
      input:focus { border-color:#50a99c !important; box-shadow:0 0 0 2px rgba(8,127,115,.12) !important; }
      div.stButton > button, [data-testid="stFormSubmitButton"] button { height:46px; min-width:148px; white-space:nowrap; border:0; border-radius:10px; background:#087f73; color:#fff; font-weight:700; padding:0 18px; transition:background .15s ease,transform .15s ease; }
      div.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background:#066b62; color:#fff; border:0; transform:translateY(-1px); }
      .pl-helper { color:var(--pl-muted); font-size:12px; margin-top:5px; }
      .pl-result { background:#fff; border:1px solid var(--pl-line); border-radius:17px; padding:22px 24px; box-shadow:0 8px 26px rgba(27,53,70,.045); }
      .pl-result-top { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; }
      .pl-label { color:var(--pl-muted); font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; }
      .pl-scoreline { display:flex; align-items:baseline; gap:10px; margin-top:3px; }
      .pl-score { color:var(--pl-ink); font:800 36px/1.2 'Manrope',sans-serif; letter-spacing:-1.4px; }
      .pl-threshold { color:var(--pl-muted); font-size:12px; }
      .pl-pill { display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:100px; font-size:12px; font-weight:700; white-space:nowrap; }
      .pl-pill-danger { color:var(--pl-red); background:var(--pl-red-soft); }
      .pl-pill-safe { color:var(--pl-teal); background:var(--pl-soft); }
      .pl-dot { width:8px;height:8px;border-radius:50%;background:currentColor; }
      .pl-meter { height:8px; margin-top:13px; background:#e9eef0; border-radius:99px; overflow:hidden; }
      .pl-meter-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#e5a25b,#b64f40); }
      .pl-section-title { color:var(--pl-ink); font:700 17px 'Manrope',sans-serif; letter-spacing:-.35px; margin:24px 0 4px; }
      .pl-section-note { color:var(--pl-muted); font-size:12px; margin:0 0 10px; }
      .pl-table { width:100%; border-collapse:collapse; font-size:13px; }
      .pl-table th { color:var(--pl-muted); text-align:left; font-size:10px; letter-spacing:.85px; text-transform:uppercase; padding:10px 9px; border-bottom:1px solid var(--pl-line); }
      .pl-table td { color:var(--pl-ink); padding:11px 9px; border-bottom:1px solid #edf1f3; vertical-align:top; }
      .pl-table tr:last-child td { border-bottom:0; }
      .pl-table td:last-child,.pl-table th:last-child { text-align:right; }
      .pl-contrib-up { color:#a34835; font-weight:700; }
      .pl-contrib-down { color:#087f73; font-weight:700; }
      .pl-warning { background:#fff6ec; border:1px solid #f3e0c6; border-radius:11px; color:#76552f; font-size:12px; line-height:1.55; padding:11px 13px; margin-top:16px; }
      .pl-footer { text-align:center; color:#7c8b97; font-size:11px; padding-top:20px; }
      @media(max-width:640px) {
        .block-container { padding: .8rem 1rem 2.5rem; }
        .pl-topnote { display:none; }
        [data-testid="stForm"] { padding:16px; }
        .pl-result { padding:17px 15px; }
        .pl-result-top { flex-direction:column; }
        .pl-table { font-size:12px; }
        .pl-table th,.pl-table td { padding:9px 5px; }
      }
    </style>
    <div class="pl-topbar"><div class="pl-brand"><span class="pl-logo">⌁</span>PhishLens</div><div class="pl-topnote">URL review prototype&nbsp;&nbsp;·&nbsp;&nbsp;Machine learning</div></div>
    <section class="pl-hero"><div class="pl-eyebrow">Website security · ML review</div><h1 class="pl-title">Check a website URL</h1><p class="pl-subtitle">Review URL patterns, estimate phishing risk, and see which features influenced the model's prediction.</p></section>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.info("The trained model is missing. Run `python train_model.py` in the VS Code terminal first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)

with st.form("url_review", clear_on_submit=False):
    input_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
    with input_col:
        url = st.text_input("Website URL", placeholder="https://example.com/sign-in", label_visibility="visible")
    with button_col:
        submitted = st.form_submit_button("Review URL", type="primary", use_container_width=True)
    st.markdown('<div class="pl-helper">PhishLens checks URL text only. It does not open the website or inspect its page.</div>', unsafe_allow_html=True)

if submitted:
    if not url.strip():
        st.warning("Enter a URL to review.")
        st.stop()

    features = extract_many([url])[bundle["feature_names"]]
    phishing_score = float(bundle["model"].predict_proba(features)[0, 1])
    flagged = phishing_score >= bundle["threshold"]
    score_pct = phishing_score * 100
    threshold_pct = bundle["threshold"] * 100
    pill_class = "pl-pill-danger" if flagged else "pl-pill-safe"
    verdict = "Potential phishing" if flagged else "Not flagged by model"
    bar_color = "#b64f40" if flagged else "#087f73"

    st.markdown(
        f"""
        <section class="pl-result" aria-label="Phishing URL prediction">
          <div class="pl-result-top">
            <div><div class="pl-label">Phishing risk score</div>
              <div class="pl-scoreline"><span class="pl-score">{score_pct:.1f}%</span><span class="pl-threshold">Alert threshold: {threshold_pct:.1f}%</span></div>
            </div>
            <div class="pl-pill {pill_class}"><span class="pl-dot"></span>{verdict}</div>
          </div>
          <div class="pl-meter" role="img" aria-label="Phishing score {score_pct:.1f} percent"><div class="pl-meter-fill" style="width:{score_pct:.1f}%;background:{bar_color}"></div></div>
          <div class="pl-section-title">Why the model scored it this way</div>
          <p class="pl-section-note">SHAP estimates how each URL feature moved the phishing score.</p>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Calculating feature contributions…"):
        explainer = shap.Explainer(bundle["model"].predict_proba, bundle["background"], algorithm="permutation")
        explanation = explainer(features, max_evals=2 * len(bundle["feature_names"]) + 1)
        values = explanation.values
        contributions = values[0, :, 1] if values.ndim == 3 else values[0]
        detail = pd.DataFrame({
            "feature": bundle["feature_names"],
            "contribution": contributions * 100,
            "value": features.iloc[0].to_numpy(),
        }).sort_values("contribution", key=lambda col: col.abs(), ascending=False).head(8)

    rows = []
    for _, row in detail.iterrows():
        contribution = float(row["contribution"])
        direction = "Increases risk" if contribution >= 0 else "Lowers risk"
        css = "pl-contrib-up" if contribution >= 0 else "pl-contrib-down"
        rows.append(
            "<tr><td>{}</td><td>{}</td><td class='{}'>{} {:+.1f} pp</td></tr>".format(
                escape(str(row["feature"])), escape(str(row["value"])), css, direction, contribution
            )
        )
    table = (
        "<table class='pl-table'><thead><tr><th>URL feature</th><th>This URL</th><th>Score effect</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    st.markdown(table, unsafe_allow_html=True)
    st.markdown(
        "<div class='pl-warning'><b>Keep in mind:</b> A model prediction is only a signal. A URL marked 'not flagged' is not guaranteed to be safe.</div></section>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"<div class='pl-footer'>Model: {escape(str(bundle['model_name']))} &nbsp;·&nbsp; URL features only &nbsp;·&nbsp; No websites are opened or crawled</div>",
    unsafe_allow_html=True,
)
