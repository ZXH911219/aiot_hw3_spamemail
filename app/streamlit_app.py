"""Streamlit demo app for Spam Classification

Features:
- Load dataset (default: data/processed/sms_spam_clean.csv) or upload CSV
- Show class distribution and top tokens per class
- If models are present in `models/`, allow single-text prediction and show metrics (confusion matrix, ROC/PR)

Run locally:
    streamlit run app/streamlit_app.py

Note: model artifacts are expected at `models/vectorizer.joblib`, `models/model.joblib`, `models/labels.joblib`.
"""
from __future__ import annotations

import os
from typing import Optional

import altair as alt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, confusion_matrix)


DEFAULT_DATA = "data/processed/sms_spam_clean.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def top_tokens_by_class(texts: pd.Series, labels: pd.Series, topn: int = 20):
    cv = CountVectorizer(ngram_range=(1, 1), stop_words="english")
    X = cv.fit_transform(texts)
    tokens = cv.get_feature_names_out()
    df = pd.DataFrame(X.toarray(), columns=tokens)
    results = {}
    for cls in sorted(labels.unique()):
        idx = labels == cls
        s = df[idx].sum(axis=0).sort_values(ascending=False).head(topn)
        results[cls] = s
    return results


def load_model_if_exists(models_dir: str = "models") -> Optional[dict]:
    vec_path = os.path.join(models_dir, "vectorizer.joblib")
    model_path = os.path.join(models_dir, "model.joblib")
    labels_path = os.path.join(models_dir, "labels.joblib")
    if os.path.exists(vec_path) and os.path.exists(model_path) and os.path.exists(labels_path):
        vec = joblib.load(vec_path)
        clf = joblib.load(model_path)
        labels = joblib.load(labels_path)
        return {"vec": vec, "clf": clf, "labels": labels}
    return None


def plot_class_distribution(df: pd.DataFrame, label_col: str):
    counts = df[label_col].value_counts().reset_index()
    counts.columns = [label_col, "count"]
    chart = alt.Chart(counts).mark_bar().encode(x=label_col, y="count:Q", tooltip=[label_col, "count"])
    st.altair_chart(chart, use_container_width=True)


def show_token_tables(results: dict):
    for cls, series in results.items():
        st.write(f"Top tokens for: {cls}")
        st.table(series.reset_index().rename(columns={"index": "token", 0: "count"}))


def main():
    st.set_page_config(page_title="Spam Classifier Demo", layout="wide")
    st.title("Spam Classification Demo")

    with st.sidebar:
        st.header("Data & Models")
        data_source = st.radio("Data source", ["Default dataset", "Upload CSV"])
        uploaded = None
        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
        models_dir = st.text_input("Models directory", value="models")
        st.markdown("---")
        st.write("Model artifacts loader will look for `vectorizer.joblib`, `model.joblib`, `labels.joblib` in the models dir.")

    # Load data
    df = None
    if data_source == "Default dataset":
        if os.path.exists(DEFAULT_DATA):
            df = load_data(DEFAULT_DATA)
        else:
            st.warning(f"Default dataset not found at `{DEFAULT_DATA}`. Upload a CSV or run preprocessing.")
    else:
        if uploaded is not None:
            df = pd.read_csv(uploaded)

    if df is None:
        st.info("No dataset loaded yet. Use the sidebar to upload or vendor the default dataset and run preprocessing/training.")
        st.stop()

    # Choose columns
    cols = list(df.columns)
    col1, col2 = st.columns(2)
    with col1:
        label_col = st.selectbox("Label column", options=cols, index=0)
    with col2:
        text_col = st.selectbox("Text column", options=cols, index=min(1, len(cols)-1))

    st.header("Data overview")
    st.write(df.head())

    st.subheader("Class distribution")
    plot_class_distribution(df, label_col)

    st.subheader("Top tokens per class (basic)")
    topn = st.slider("Top N tokens", 5, 50, 20)
    token_results = top_tokens_by_class(df[text_col].astype(str), df[label_col].astype(str), topn)
    # Show as small tables
    for cls, series in token_results.items():
        st.write(f"Top tokens for {cls}")
        st.dataframe(series.head(topn).reset_index().rename(columns={"index": "token", 0: "count"}))

    # Load models if available
    model_bundle = load_model_if_exists(models_dir)
    st.sidebar.markdown("---")
    if model_bundle:
        st.sidebar.success("Model artifacts found")
    else:
        st.sidebar.warning("No model artifacts found in models dir")

    if model_bundle:
        st.header("Prediction")
        sample_text = st.text_area("Enter text to classify", value="Free entry in 2 a wkly comp to win cash", height=120)
        threshold = st.slider("Probability threshold for spam", 0.0, 1.0, 0.5)
        if st.button("Predict"):
            vec = model_bundle["vec"]
            clf = model_bundle["clf"]
            labels = list(model_bundle["labels"])
            x = vec.transform([sample_text])
            probs = clf.predict_proba(x)[0]
            pred_idx = int(probs.argmax())
            pred_label = labels[pred_idx]
            st.write({"predicted_label": pred_label, "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))}})

        st.header("Model evaluation on loaded dataset")
        if st.button("Run evaluation (on loaded dataset)"):
            vec = model_bundle["vec"]
            clf = model_bundle["clf"]
            Xv = vec.transform(df[text_col].astype(str))
            y_true = df[label_col].astype(str)
            y_pred = clf.predict(Xv)
            cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y_true.unique()), yticklabels=sorted(y_true.unique()))
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            try:
                RocCurveDisplay.from_estimator(clf, Xv, y_true == sorted(y_true.unique())[1])
                st.pyplot()
            except Exception:
                st.info("ROC curve unavailable for non-binary or unsupported label encoding")

            try:
                PrecisionRecallDisplay.from_estimator(clf, Xv, y_true == sorted(y_true.unique())[1])
                st.pyplot()
            except Exception:
                st.info("PR curve unavailable for non-binary or unsupported label encoding")

    # Reports: show generated visualization artifacts if they exist
    st.markdown("---")
    st.header("Saved reports & visualizations")
    reports_dir = os.path.join("reports", "visualizations")
    if os.path.exists(reports_dir):
        imgs = [
            ("Class distribution", os.path.join(reports_dir, "class_distribution.png")),
            ("Top tokens (ham)", os.path.join(reports_dir, "top_tokens_ham.png")),
            ("Top tokens (spam)", os.path.join(reports_dir, "top_tokens_spam.png")),
            ("Confusion matrix", os.path.join(reports_dir, "confusion_matrix.png")),
            ("ROC curve", os.path.join(reports_dir, "roc_curve.png")),
            ("Precision-Recall curve", os.path.join(reports_dir, "pr_curve.png")),
            ("Threshold sweep (f1)", os.path.join(reports_dir, "threshold_sweep.png")),
        ]
        for title, path in imgs:
            if os.path.exists(path):
                st.subheader(title)
                st.image(path, use_column_width=True)

        # Show threshold sweep CSV if present
        thresh_csv = os.path.join(reports_dir, "threshold_sweep.csv")
        if os.path.exists(thresh_csv):
            st.subheader("Threshold sweep (CSV)")
            try:
                df_thresh = pd.read_csv(thresh_csv)
                st.dataframe(df_thresh)
            except Exception:
                st.write("Unable to load threshold sweep CSV")

        # Provide links to raw CSVs for top tokens
        st.subheader("Top tokens (CSV)")
        for cls in ["ham", "spam"]:
            csv_path = os.path.join(reports_dir, f"top_tokens_{cls}.csv")
            if os.path.exists(csv_path):
                st.write(f"{cls}")
                st.download_button(label=f"Download top_tokens_{cls}.csv", data=open(csv_path, "rb"), file_name=f"top_tokens_{cls}.csv")
    else:
        st.info("No saved reports found. Run the visualization script `scripts/visualize_spam.py` after training to generate reports.")

    st.caption("This demo app is scaffolded from the example project and runs locally. To deploy, push this repo to GitHub and use Streamlit Cloud or another hosting service.")


if __name__ == "__main__":
    main()
