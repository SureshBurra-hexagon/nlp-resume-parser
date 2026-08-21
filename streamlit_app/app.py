from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.advanced_resume_classifier import AdvancedResumeClassifier
from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.utils.resume_parser import parse_resume

st.set_page_config(page_title="NLP Resume Parser", layout="wide")
st.title("NLP Resume Parser - Phase 3 Demo")

resume_text = st.text_area("Paste resume text", height=240)

baseline_model_dir = Path(__file__).resolve().parents[1] / "models" / "baseline"
advanced_model_dir = Path(__file__).resolve().parents[1] / "models" / "advanced"
model_choice = st.radio("Model", ["baseline", "advanced"], horizontal=True)


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

if st.button("Parse Resume"):
    if not resume_text.strip():
        st.warning("Please paste resume text first.")
    else:
        parsed = parse_resume(resume_text)
        st.subheader("Extracted Information")
        st.json(parsed)

        baseline_artifacts_available = (baseline_model_dir / "tfidf.joblib").exists() and (baseline_model_dir / "classifier.joblib").exists()
        advanced_artifacts_available = (advanced_model_dir / "advanced_pipeline.joblib").exists()

        if model_choice == "baseline" and baseline_artifacts_available:
            extractor, classifier = load_artifacts(str(baseline_model_dir))
            features = extractor.transform([parsed["normalized_text"]])
            prediction = classifier.predict(features)[0]
            confidence = float(max(classifier.predict_proba(features)[0]))
            st.subheader("Predicted Profile Category")
            st.success(f"{prediction} ({confidence:.2%})")
        elif model_choice == "advanced" and advanced_artifacts_available:
            classifier = load_advanced_artifacts(str(advanced_model_dir))
            probabilities = classifier.predict_proba([parsed["normalized_text"]])[0]
            prediction = classifier.model.classes_[probabilities.argmax()]
            confidence = float(max(probabilities))
            st.subheader("Predicted Profile Category")
            st.success(f"{prediction} ({confidence:.2%})")
        else:
            if model_choice == "advanced":
                st.info("Train the advanced model first: python scripts/train_advanced.py")
            else:
                st.info("Train the baseline model first: python scripts/train.py")
