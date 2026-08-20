# NLP Resume Parser

Baseline implementation of an academic NLP resume parsing project, plus a comprehensive advanced enhancement roadmap.

## Implemented baseline

### Core pipeline
- Text preprocessing and normalization (`src/preprocessing/text_cleaner.py`)
- Contact entity extraction (email, phone)
- Skill extraction and resume parsing utility (`src/utils/resume_parser.py`)
- TF-IDF feature extraction (`src/feature_extraction/tfidf_features.py`)
- Resume profile classification with Logistic Regression (`src/models/resume_classifier.py`)
- Evaluation metrics: Accuracy, Precision, Recall, F1 (`src/evaluation/metrics.py`)

### Scripts and demo
- Training script: `python scripts/train.py`
- Evaluation script: `python scripts/evaluate.py`
- Streamlit demo: `streamlit run streamlit_app/app.py`

### Sample data and tests
- Sample train/test datasets in `data/sample_resumes/`
- Unit tests in `tests/unit/`

## Installation

```bash
pip install -r requirements-dev.txt
```

## Quick start

```bash
python scripts/train.py
python scripts/evaluate.py
pytest -q
streamlit run streamlit_app/app.py
```

## Advanced roadmap

For the full advanced project scope (transformers, multimodal parsing, optimization, explainability, API, deployment, benchmarking, and research deliverables), see:

- [`docs/ADVANCED_ENHANCEMENT_PLAN.md`](docs/ADVANCED_ENHANCEMENT_PLAN.md)
