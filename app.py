"""EcoTrace: interactive evaluation of saved QSAR biodegradation classifiers."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "biodeg_class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="EcoTrace | QSAR Lab", page_icon="🧪", layout="wide")
st.markdown("""
<style>
:root { --ink: #152b35; --muted: #60747c; --accent: #0c786d; --wash: #edf4f2; }
.block-container {padding-top: 2.0rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#102f38,#17675e); padding: 1.55rem 1.8rem;
       border-radius: 18px; color: #f5fbf8; margin-bottom: 1.1rem;}
.hero h1 {font-size: 2.05rem; margin: 0 0 .35rem 0; color: #f5fbf8;}
.hero p {margin: 0; color: #d5e9e4; font-size: .98rem;}
.kicker {letter-spacing: .12em; text-transform: uppercase; font-size: .7rem; color: #a8d8cc;}
.section-label {color: #0c786d; text-transform: uppercase; letter-spacing: .1em;
                font-size: .72rem; font-weight: 700; margin-top: .6rem;}
div[data-testid="stMetric"] {background: #f1f6f4; border: 1px solid #d8e5e1;
  padding: .8rem 1rem; border-radius: 12px;}
div[data-testid="stMetricLabel"] {color: #60747c;}
.small-note {color: #60747c; font-size: .86rem;}
</style>
<div class="hero">
  <div class="kicker">EcoTrace · Molecular descriptor intelligence</div>
  <h1>Can this molecule biodegrade readily?</h1>
  <p>Compare five classical classifiers on an unseen QSAR test file. Models are trained offline;
  this app only scores the uploaded labelled test data.</p>
</div>
""", unsafe_allow_html=True)


def load_assets():
    feature_path = MODEL_DIR / "feature_names.json"
    if not feature_path.exists():
        raise FileNotFoundError("Model assets are missing. Run `python train.py` first.")
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    models = {}
    missing = []
    for display_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            missing.append(filename)
        else:
            models[display_name] = joblib.load(path)
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing) + ". Run `python train.py`.")
    return features, models


def normalise_labels(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.upper()
    aliases = {
        "READY": "RB", "READY BIODEGRADABLE": "RB",
        "NOT READY": "NRB", "NOT READY BIODEGRADABLE": "NRB",
        "NON-READY": "NRB", "NON READY": "NRB",
        "0": "NRB", "1": "RB",
    }
    cleaned = cleaned.replace(aliases)
    if not set(cleaned.unique()).issubset({"RB", "NRB"}):
        raise ValueError("Target must contain only RB/NRB (or 1/0) labels.")
    return cleaned.map({"NRB": 0, "RB": 1}).astype(int)


def evaluate_one(model, X, y):
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    pos_idx = int(np.where(np.asarray(model.classes_) == 1)[0][0])
    positive_probability = probabilities[:, pos_idx]
    return {
        "prediction": prediction,
        "probability": positive_probability,
        "Accuracy": accuracy_score(y, prediction),
        "AUC": roc_auc_score(y, positive_probability),
        "Precision": precision_score(y, prediction, pos_label=1, zero_division=0),
        "Recall": recall_score(y, prediction, pos_label=1, zero_division=0),
        "F1": f1_score(y, prediction, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
    }


try:
    feature_names, saved_models = load_assets()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.markdown('<div class="section-label">01 / Test-set input</div>', unsafe_allow_html=True)
st.write("Upload a labelled CSV containing all 41 molecular descriptors and the **biodeg_class** target (`RB` or `NRB`). The bundled holdout file is used automatically when no file is uploaded.")
upload = st.file_uploader("Choose test CSV", type=["csv"], help="Do not upload training data. Keep the target column so the app can calculate evaluation metrics.")

if upload is not None:
    try:
        test_df = pd.read_csv(upload)
        input_origin = f"Uploaded file: {upload.name}"
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        st.stop()
else:
    default_path = ROOT / "test_data.csv"
    if not default_path.exists():
        st.info("No test file is bundled yet. Run `python train.py` to generate the stratified holdout CSV.")
        st.stop()
    test_df = pd.read_csv(default_path)
    input_origin = "Project holdout file: test_data.csv"

# Accept a common source target-column alias while keeping the internal output schema fixed.
target_aliases = [TARGET, "class", "experimental class", "experimental_class"]
target_col = next((c for c in target_aliases if c in test_df.columns), None)
if target_col is None:
    st.error(f"Target column missing. Expected `{TARGET}` (or source column `class`).")
    st.stop()
missing_features = [col for col in feature_names if col not in test_df.columns]
if missing_features:
    st.error(f"Missing {len(missing_features)} required descriptor columns: {', '.join(missing_features[:8])}" + ("…" if len(missing_features) > 8 else ""))
    st.stop()

try:
    X_eval = test_df[feature_names].apply(pd.to_numeric, errors="raise")
    y_eval = normalise_labels(test_df[target_col])
    if X_eval.isna().any().any() or y_eval.isna().any():
        raise ValueError("The uploaded CSV contains missing predictor or target values.")
    if len(test_df) < 2:
        raise ValueError("Upload at least two labelled records for evaluation.")
    if y_eval.nunique() != 2:
        raise ValueError("The uploaded test file must contain both RB and NRB classes to calculate AUC.")
except Exception as exc:
    st.error(f"Test CSV validation failed: {exc}")
    st.stop()

st.caption(f"{input_origin} · {len(test_df):,} rows · {len(feature_names)} predictors · target: `{target_col}`")
st.markdown('<div class="section-label">02 / Model comparison</div>', unsafe_allow_html=True)
results = {name: evaluate_one(model, X_eval, y_eval) for name, model in saved_models.items()}
metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
comparison = pd.DataFrame([
    {"Model": name, **{metric: values[metric] for metric in metric_cols}}
    for name, values in results.items()
]).sort_values(["MCC", "F1"], ascending=False).reset_index(drop=True)

st.dataframe(
    comparison.style.format({metric: "{:.4f}" for metric in metric_cols})
      .highlight_max(subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], color="#d6eee7"),
    use_container_width=True, hide_index=True,
)
winner = comparison.iloc[0]
st.caption(f"Current test-file leader by MCC: **{winner['Model']}** (MCC {winner['MCC']:.3f}). This is a test-set comparison, not a guarantee of future performance.")

st.markdown('<div class="section-label">03 / Model inspection</div>', unsafe_allow_html=True)
selected_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()), index=list(MODEL_FILES.keys()).index(winner["Model"]))
selected = results[selected_name]
cols = st.columns(6)
for col, metric in zip(cols, metric_cols):
    col.metric(metric, f"{selected[metric]:.3f}")

left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_eval, selected["prediction"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.1, 4.0))
    image = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    ax.set_title("Actual vs predicted")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks([0, 1], ["NRB", "RB"])
    ax.set_yticks([0, 1], ["NRB", "RB"])
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "#173b33", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y_eval, selected["probability"])
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots(figsize=(5.1, 4.0))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax2.set(xlabel="False positive rate", ylabel="True positive rate", title="RB as positive class", xlim=(0, 1), ylim=(0, 1.02))
    ax2.legend(loc="lower right")
    ax2.grid(alpha=.2)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with st.expander("Classification report"):
    report = classification_report(
        y_eval, selected["prediction"], labels=[0, 1],
        target_names=["NRB · not ready", "RB · ready"], output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True)

st.markdown('<div class="section-label">04 / Prediction audit</div>', unsafe_allow_html=True)
audit = test_df.copy()
audit["actual_class"] = y_eval.map({0: "NRB", 1: "RB"})
audit["predicted_class"] = pd.Series(selected["prediction"]).map({0: "NRB", 1: "RB"}).values
audit["P(RB)"] = selected["probability"]
audit["correct"] = audit["actual_class"] == audit["predicted_class"]
st.dataframe(audit[["actual_class", "predicted_class", "P(RB)", "correct"]].head(30), use_container_width=True, hide_index=True)
st.download_button(
    "Download scored test rows",
    data=audit.to_csv(index=False).encode("utf-8"),
    file_name="ecotrace_scored_test_rows.csv",
    mime="text/csv",
)
st.markdown("---")
st.markdown("<p class='small-note'>Research/teaching demonstration only. QSAR descriptors are not a substitute for laboratory biodegradation testing or regulatory assessment. Evaluation metrics describe only the labelled CSV supplied in this session.</p>", unsafe_allow_html=True)
