from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.advanced_resume_classifier import AdvancedResumeClassifier
from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.optimization.inference import ResumeBatchProcessor

st.set_page_config(page_title="NLP Resume Parser", layout="wide")
st.title("NLP Resume Parser - Phase 4 Web App")

baseline_model_dir = Path(__file__).resolve().parents[1] / "models" / "baseline"
advanced_model_dir = Path(__file__).resolve().parents[1] / "models" / "advanced"
processor = ResumeBatchProcessor()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def load_artifacts(artifact_dir: str):
    path = Path(artifact_dir)
    extractor = TfidfFeatureExtractor.load(str(path / "tfidf.joblib"))
    classifier = ResumeClassifier.load(str(path / "classifier.joblib"))
    return extractor, classifier


@st.cache_resource
def load_advanced_artifacts(artifact_dir: str):
    path = Path(artifact_dir)
    return AdvancedResumeClassifier.load(str(path / "advanced_pipeline.joblib"))


def _predict_single(parsed: dict, model_choice: str) -> tuple[str | None, float | None, str | None]:
    baseline_artifacts_available = (baseline_model_dir / "tfidf.joblib").exists() and (baseline_model_dir / "classifier.joblib").exists()
    advanced_artifacts_available = (advanced_model_dir / "advanced_pipeline.joblib").exists()

    if model_choice == "baseline":
        if not baseline_artifacts_available:
            return None, None, "Train the baseline model first: python scripts/train.py"
        extractor, classifier = load_artifacts(str(baseline_model_dir))
        features = extractor.transform([parsed["normalized_text"]])
        prediction = str(classifier.predict(features)[0])
        confidence = float(max(classifier.predict_proba(features)[0]))
        return prediction, confidence, None

    if not advanced_artifacts_available:
        return None, None, "Train the advanced model first: python scripts/train_advanced.py"
    classifier = load_advanced_artifacts(str(advanced_model_dir))
    probabilities = classifier.predict_proba([parsed["normalized_text"]])[0]
    prediction = str(classifier.model.classes_[probabilities.argmax()])
    confidence = float(max(probabilities))
    return prediction, confidence, None


def _history_df() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame(columns=["timestamp", "model", "prediction", "confidence", "skills"])
    rows = []
    for item in st.session_state.history:
        rows.append(
            {
                "timestamp": item["timestamp"],
                "model": item["model"],
                "prediction": item["prediction"],
                "confidence": item["confidence"],
                "skills": ", ".join(item["skills"]),
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Model", ["baseline", "advanced"], horizontal=True)
    theme = st.selectbox("Theme", ["default", "minimal", "high contrast"], index=0)
    show_confidence = st.checkbox("Show confidence", value=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.success("History cleared")

if theme == "minimal":
    st.caption("Minimal view enabled")
elif theme == "high contrast":
    st.caption("High contrast view enabled")

single_tab, batch_tab, compare_tab, history_tab, analytics_tab = st.tabs(
    ["Single Resume", "Batch Resume", "Compare & Rank", "History", "Analytics"]
)

with single_tab:
    resume_text = st.text_area("Paste resume text", height=220, key="single_input")
    if st.button("Parse & Predict", key="single_run"):
        if not resume_text.strip():
            st.warning("Please paste resume text first.")
        else:
            parsed = processor.parse(resume_text)
            prediction, confidence, error = _predict_single(parsed, model_choice)
            st.subheader("Extracted Information")
            st.json(parsed)
            if error:
                st.info(error)
            else:
                if show_confidence and confidence is not None:
                    st.success(f"{prediction} ({confidence:.2%})")
                else:
                    st.success(str(prediction))
                st.session_state.history.append(
                    {
                        "timestamp": _now_iso(),
                        "model": model_choice,
                        "prediction": prediction,
                        "confidence": confidence,
                        "skills": parsed["skills"],
                    }
                )

with batch_tab:
    st.caption("Use --- as a separator between resumes.")
    batch_input = st.text_area("Paste multiple resumes", height=220, key="batch_input")
    if st.button("Run Batch Parse & Predict", key="batch_run"):
        raw_items = [item.strip() for item in batch_input.split("\n---\n") if item.strip()]
        if not raw_items:
            st.warning("Please add at least one resume.")
        else:
            progress = st.progress(0)
            rows = []
            for idx, text in enumerate(raw_items, start=1):
                parsed = processor.parse(text)
                prediction, confidence, error = _predict_single(parsed, model_choice)
                if error:
                    st.info(error)
                    break
                rows.append(
                    {
                        "rank": 0,
                        "prediction": prediction,
                        "confidence": confidence,
                        "skills": ", ".join(parsed["skills"]),
                        "experience_years": parsed["experience_years"],
                    }
                )
                st.session_state.history.append(
                    {
                        "timestamp": _now_iso(),
                        "model": model_choice,
                        "prediction": prediction,
                        "confidence": confidence,
                        "skills": parsed["skills"],
                    }
                )
                progress.progress(idx / len(raw_items))
                time.sleep(0.02)

            if rows:
                df = pd.DataFrame(rows)
                df = df.sort_values(by="confidence", ascending=False).reset_index(drop=True)
                df["rank"] = df.index + 1
                st.dataframe(df, use_container_width=True)

with compare_tab:
    st.caption("Compare parsed resumes by confidence and extracted features.")
    compare_df = _history_df()
    if compare_df.empty:
        st.info("No prediction history yet.")
    else:
        ranked = compare_df.sort_values(by="confidence", ascending=False, na_position="last")
        st.dataframe(ranked, use_container_width=True)

with history_tab:
    history_df = _history_df()
    if history_df.empty:
        st.info("No history available.")
    else:
        query = st.text_input("Search history by skill or prediction", value="")
        selected_model = st.selectbox("Filter by model", ["all", "baseline", "advanced"], index=0)

        filtered = history_df
        if selected_model != "all":
            filtered = filtered[filtered["model"] == selected_model]
        if query.strip():
            q = query.strip().lower()
            filtered = filtered[
                filtered["skills"].str.lower().str.contains(q) | filtered["prediction"].str.lower().str.contains(q)
            ]

        st.dataframe(filtered, use_container_width=True)
        st.download_button(
            label="Export history as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="resume_parser_history.csv",
            mime="text/csv",
        )

with analytics_tab:
    history_df = _history_df()
    if history_df.empty:
        st.info("Run predictions to see analytics.")
    else:
        confidence_values = history_df["confidence"].dropna()
        avg_conf = float(confidence_values.mean()) if not confidence_values.empty else 0.0
        st.metric("Total Predictions", int(len(history_df)))
        st.metric("Average Confidence", f"{avg_conf:.2%}")

        all_skills = []
        for skills in history_df["skills"].tolist():
            all_skills.extend([skill.strip() for skill in skills.split(",") if skill.strip()])

        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts().head(10)
            st.bar_chart(skill_counts)
        else:
            st.info("No skills extracted yet.")
