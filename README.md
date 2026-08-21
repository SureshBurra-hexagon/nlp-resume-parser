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

### Phase 1 transformer module
- Hugging Face transformer text embeddings with mean pooling (`src/embedding/transformer_features.py`)
- Transformer embedding + Logistic Regression classifier (`src/models/transformer_resume_classifier.py`)
- Transformer training/evaluation scripts (`scripts/train_transformer.py`, `scripts/evaluate_transformer.py`)

### Phase 2 multimodal ingestion module
- File-based ingestion for TXT, HTML, DOCX, and PDF resumes (`src/ingestion/file_text_extractor.py`)
- File-to-structured parsing helper (`src/utils/resume_parser.py::parse_resume_file`)
- CLI parser for resume files (`scripts/parse_file.py`)

### Phase 3 advanced modules
- Section-aware and ontology-aware resume parsing with education, certification, and experience extraction (`src/ner/resume_entities.py`)
- Hybrid word + character TF-IDF embeddings (`src/embedding/hybrid_features.py`)
- Deterministic training-time augmentation for synthetic resume variants (`src/augmentation/resume_augmenter.py`)
- Soft-voting advanced ensemble classifier (`src/models/advanced_resume_classifier.py`)
- Hyperparameter search and batch inference utilities (`src/optimization/`)

### Phase 4 web, API, and deployment modules
- Streamlit phase 4 experience with single and batch workflows, history, ranking, export, and analytics (`streamlit_app/app.py`)
- FastAPI phase 4 backend with parse/predict/evaluate/analyze endpoints, optional bearer/JWT auth, CORS, rate limiting, and structured error handling (`fastapi_app/main.py`)
- Container deployment assets: `Dockerfile`, `docker-compose.yml`, `.env.example`, and runbook (`docs/DEPLOYMENT_RUNBOOK.md`)

### Phase 5 advanced product features
- ATS keyword matching and scoring against a job description (`src/analysis/ats_scorer.py`)
- Experience level classification and career path projection (`src/analysis/experience_classifier.py`)
- Multi-dimensional resume quality scoring and salary band estimation (`src/analysis/resume_scorer.py`)
- Recommendation engine for actionable resume improvement suggestions (`src/analysis/recommender.py`)

### Scripts and demo
- Training script: `python scripts/train.py`
- Evaluation script: `python scripts/evaluate.py`
- Transformer training script: `python scripts/train_transformer.py`
- Transformer evaluation script: `python scripts/evaluate_transformer.py`
- File parse script: `python scripts/parse_file.py --file /absolute/path/to/resume.pdf`
- Advanced training script: `python scripts/train_advanced.py`
- Advanced evaluation/benchmark script: `python scripts/evaluate_advanced.py`
- Streamlit demo: `streamlit run streamlit_app/app.py`
- FastAPI demo: `uvicorn fastapi_app.main:app --reload`

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
python scripts/train_transformer.py
python scripts/evaluate_transformer.py
python scripts/parse_file.py --file /absolute/path/to/resume.txt
python scripts/train_advanced.py
python scripts/evaluate_advanced.py
pytest -q
streamlit run streamlit_app/app.py
uvicorn fastapi_app.main:app --reload
docker compose up --build
```

## FastAPI endpoints

- `GET /health` - API health with baseline and advanced artifact availability
- `POST /parse` - parse resume text into normalized text, contacts, skills, sections, education, certifications, and experience
- `POST /predict` - parse resume text and predict the baseline profile category
- `POST /predict/advanced` - parse resume text and predict with the phase 3 advanced ensemble
- `POST /predict/advanced/batch` - batch advanced predictions using cached parsing
- `POST /evaluate` - evaluate classification labels/predictions with core metrics
- `POST /analyze` - analyze resume batches for aggregate parsing insights

## Advanced roadmap

For the full advanced project scope (transformers, multimodal parsing, optimization, explainability, API, deployment, benchmarking, and research deliverables), see:

- [`docs/ADVANCED_ENHANCEMENT_PLAN.md`](docs/ADVANCED_ENHANCEMENT_PLAN.md)
