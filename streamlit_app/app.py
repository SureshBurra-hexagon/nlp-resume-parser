from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.utils.resume_parser import parse_resume

st.set_page_config(page_title="NLP Resume Parser", layout="wide")
st.title("NLP Resume Parser - Baseline Demo")

resume_text = st.text_area("Paste resume text", height=240)

model_dir = Path("models/baseline")
artifacts_available = (model_dir / "tfidf.joblib").exists() and (model_dir / "classifier.joblib").exists()

if st.button("Parse Resume"):
    if not resume_text.strip():
        st.warning("Please paste resume text first.")
    else:
        parsed = parse_resume(resume_text)
        st.subheader("Extracted Information")
        st.json(parsed)

        if artifacts_available:
            extractor = TfidfFeatureExtractor.load(str(model_dir / "tfidf.joblib"))
            classifier = ResumeClassifier.load(str(model_dir / "classifier.joblib"))
            prediction = classifier.predict(extractor.transform([parsed["normalized_text"]]))[0]
            st.subheader("Predicted Profile Category")
            st.success(prediction)
        else:
            st.info("Train the model first: python scripts/train.py")
