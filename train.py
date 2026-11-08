"""Train five baseline classifiers for the UCI QSAR biodegradation dataset.

Run from the project root:
    python train.py

The script fetches the official UCI dataset, creates one stratified holdout split,
fits all five classifiers on the same training rows, and writes the test CSV,
models, metrics, and a concise generated results narrative.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
TARGET = "biodeg_class"
POSITIVE_LABEL = "RB"
NEGATIVE_LABEL = "NRB"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Canonical names from the UCI QSAR biodegradation dataset description.
FEATURE_NAMES = [
    "SpMax_L", "J_Dz_e", "nHM", "F01_N_N", "F04_C_N", "NssssC",
    "nCb_sub", "C_percent", "nCp", "nO", "F03_C_N", "SdssC",
    "HyWi_B_m", "LOC", "SM6_L", "F03_C_O", "Me", "Mi", "nN_N",
    "nArNO2", "nCRX3", "SpPosA_B_p", "nCIR", "B01_C_Br", "B03_C_Cl",
    "N_073", "SpMax_A", "Psi_i_1d", "B04_C_Br", "SdO", "TI2_L",
    "nCrt", "C_026", "F02_C_N", "nHDon", "SpMax_B_m", "Psi_i_A",
    "nN", "SM6_B_m", "nArCOOR", "nX",
]


def _normalise_target(values: pd.Series) -> pd.Series:
    """Normalise documented UCI labels to NRB/RB without guessing unknown codes."""
    cleaned = values.astype(str).str.strip().str.upper()
    aliases = {
        "READY": "RB",
        "READY BIODEGRADABLE": "RB",
        "NOT READY": "NRB",
        "NOT READY BIODEGRADABLE": "NRB",
        "NON-READY": "NRB",
        "NON READY": "NRB",
    }
    cleaned = cleaned.replace(aliases)
    found = set(cleaned.unique())
    if found != {"RB", "NRB"}:
        raise ValueError(
            "Unexpected target labels from UCI. Expected RB/NRB; received: "
            f"{sorted(found)}. Inspect the downloaded target column before proceeding."
        )
    return cleaned


def fetch_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch official UCI dataset with a direct CSV fallback."""
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=254)
        X = dataset.data.features.copy()
        y_frame = dataset.data.targets.copy()
        y = y_frame.iloc[:, 0].copy() if isinstance(y_frame, pd.DataFrame) else pd.Series(y_frame)
        if X.shape[1] != 41:
            raise ValueError(f"UCI API returned {X.shape[1]} predictors; expected 41.")
        X.columns = FEATURE_NAMES
        return X, _normalise_target(pd.Series(y, name=TARGET))
    except Exception as api_error:
        # UCI's legacy file uses semicolon delimiters and has no header row.
        fallback_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"
        )
        try:
            raw = pd.read_csv(fallback_url, sep=";", header=None)
            if raw.shape[1] != 42:
                raise ValueError(
                    f"Fallback CSV has {raw.shape[1]} columns; expected 41 predictors + target."
                )
            X = raw.iloc[:, :41].copy()
            X.columns = FEATURE_NAMES
            y = raw.iloc[:, 41].copy()
            print("Note: fetched via UCI legacy CSV endpoint after API attempt:", api_error)
            return X, _normalise_target(y)
        except Exception as fallback_error:
            raise RuntimeError(
                "Could not fetch the UCI QSAR dataset. Check BITS Lab internet access and "
                "install ucimlrepo. Original API error: " + str(api_error) +
                "; fallback error: " + str(fallback_error)
            ) from fallback_error


def build_models() -> Dict[str, ClassifierMixin | Pipeline]:
    """Return the five assignment classifiers with leakage-safe preprocessing."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0, class_weight="balanced", solver="liblinear", max_iter=3000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=5, min_samples_leaf=4,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance", p=2)),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", GaussianNB(var_smoothing=1e-9)),
        ]),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", SVC(
                C=1.5, kernel="rbf", probability=True, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def positive_probability(model: ClassifierMixin | Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return probability for label 1 (RB), respecting the estimator class order."""
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    positive_index = int(np.where(classes == 1)[0][0])
    return probabilities[:, positive_index]


def metric_row(model_name: str, model: ClassifierMixin | Pipeline,
               X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    prediction = model.predict(X_test)
    probability = positive_probability(model, X_test)
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, prediction),
        "AUC": roc_auc_score(y_test, probability),
        "Precision": precision_score(y_test, prediction, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, prediction, pos_label=1, zero_division=0),
        "F1": f1_score(y_test, prediction, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_test, prediction),
    }


def describe_performance(row: pd.Series) -> str:
    """Generate brief, metric-grounded observations for the assignment README."""
    precision, recall = float(row["Precision"]), float(row["Recall"])
    auc, mcc, f1 = float(row["AUC"]), float(row["MCC"]), float(row["F1"])
    balance = (
        "precision and recall are relatively balanced"
        if abs(precision - recall) < 0.08
        else ("recall is stronger than precision" if recall > precision else "precision is stronger than recall")
    )
    strength = "strong" if auc >= 0.85 else ("moderate" if auc >= 0.70 else "limited")
    return (
        f"AUC {auc:.3f} indicates {strength} ranking separation; F1 is {f1:.3f} and "
        f"MCC is {mcc:.3f}. On the positive RB class, {balance} "
        f"(precision {precision:.3f}, recall {recall:.3f})."
    )


def write_results_document(metrics: pd.DataFrame, y_test: pd.Series,
                            predictions: dict[str, np.ndarray]) -> None:
    table = metrics.copy()
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        table[col] = table[col].map(lambda value: f"{value:.4f}")
    winner = metrics.sort_values(["MCC", "F1", "AUC"], ascending=False).iloc[0]
    lines = [
        "# Generated experiment results",
        "",
        "**Evaluation protocol:** one 80:20 stratified train/test split (random seed 42). "
        "All five models use the same split; preprocessing is fitted only on training rows.",
        "**Positive class:** RB (ready biodegradable).",
        "",
        "## Test-set comparison",
        "",
        table.to_markdown(index=False),
        "",
        "## Model observations",
        "",
    ]
    for _, row in metrics.iterrows():
        lines.append(f"- **{row['Model']}:** {describe_performance(row)}")
    lines += [
        "",
        "## Overall winner",
        "",
        f"**{winner['Model']}** has the highest MCC ({winner['MCC']:.4f}) on this held-out "
        "split. MCC is used as the primary tie-breaker because the UCI class counts are "
        "unequal; confirm the choice against precision/recall and the assignment's full table.",
        "",
        "## Positive-class confusion matrices",
        "",
    ]
    for model_name, pred in predictions.items():
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("Rows = actual [NRB, RB], columns = predicted [NRB, RB].")
        lines.append("")
        lines.append(f"`[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]`")
        lines.append("")
    (RESULTS_DIR / "assignment_results.md").write_text("\n".join(lines), encoding="utf-8")



def update_readme(metrics: pd.DataFrame) -> None:
    """Populate README metric rows and the observation section after a real run."""
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        return
    content = readme_path.read_text(encoding="utf-8")
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for _, row in metrics.iterrows():
        model_name = str(row["Model"])
        values = " | ".join(f"{float(row[col]):.4f}" for col in metric_columns)
        replacement = f"| {model_name} | {values} |"
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(f"| {model_name} |"):
                lines[i] = replacement
                break
        content = "\n".join(lines) + "\n"

    start_marker = "### Observations and overall winner"
    end_marker = "## Project structure"
    if start_marker in content and end_marker in content:
        start = content.index(start_marker)
        end = content.index(end_marker, start)
        winner = metrics.sort_values(["MCC", "F1", "AUC"], ascending=False).iloc[0]
        observations = [start_marker, "", "Run-specific observations from the held-out test split:", ""]
        for _, row in metrics.iterrows():
            observations.append(f"- **{row['Model']}:** {describe_performance(row)}")
        observations += [
            "",
            f"**Overall winner:** {winner['Model']} leads by MCC ({winner['MCC']:.4f}); "
            "F1 and AUC are used as tie-breakers. Review the confusion matrix and class-wise "
            "precision/recall before interpreting this as the most useful model.",
            "",
        ]
        content = content[:start] + "\n".join(observations) + content[end:]
    readme_path.write_text(content, encoding="utf-8")

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    for folder in [MODEL_DIR, RESULTS_DIR, DATA_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    print("Fetching UCI QSAR biodegradation dataset (ID 254)...")
    X, target = fetch_dataset()
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Missing or non-numeric predictor values found; inspect the source data.")
    y = target.map({"NRB": 0, "RB": 1}).astype(int)

    if len(X) < 500 or X.shape[1] < 12:
        raise ValueError(f"Dataset fails assignment minimums: {len(X)} rows, {X.shape[1]} features.")
    if X.shape[1] != 41:
        raise ValueError(f"Expected 41 input features, received {X.shape[1]}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    test_export = X_test.copy()
    test_export[TARGET] = y_test.map({0: "NRB", 1: "RB"}).values
    test_export.to_csv(ROOT / "test_data.csv", index=False)
    X_train.to_csv(DATA_DIR / "training_features.csv", index=False)
    pd.DataFrame({TARGET: y_train.map({0: "NRB", 1: "RB"})}).to_csv(
        DATA_DIR / "training_labels.csv", index=False
    )

    print(f"Dataset: {len(X)} rows × {X.shape[1]} features")
    print("Class distribution:", target.value_counts().to_dict())
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

    models = build_models()
    metric_rows = []
    prediction_map: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        print(f"Fitting {model_name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / (model_name.lower().replace("-", "_").replace(" ", "_") + ".joblib"))
        metric_rows.append(metric_row(model_name, model, X_test, y_test))
        prediction_map[model_name] = model.predict(X_test)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    (MODEL_DIR / "feature_names.json").write_text(
        json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset": "UCI QSAR biodegradation (ID 254)",
        "target_column": TARGET,
        "positive_class": POSITIVE_LABEL,
        "negative_class": NEGATIVE_LABEL,
        "positive_label_encoding": 1,
        "negative_label_encoding": 0,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_results_document(metrics, y_test, prediction_map)
    update_readme(metrics)

    print("\nHeld-out metrics (positive class RB):")
    print(metrics.to_string(index=False, formatters={c: "{:.4f}".format for c in metrics.columns if c != "Model"}))
    print("\nCreated: test_data.csv, model/*.joblib, results/model_metrics.csv, "
          "results/assignment_results.md")
    print("Next: run `streamlit run app.py` and upload test_data.csv to verify the UI.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
