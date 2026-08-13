# NLP Resume Parser

An intelligent resume parsing system that extracts structured information
from resumes using Natural Language Processing and Machine Learning.

## Features

- **Text Extraction** – Supports PDF and plain-text resumes
- **Contact Extraction** – Email, phone, LinkedIn, GitHub via regex
- **Section Identification** – Automatically splits resumes into sections
  (Education, Experience, Skills, Projects, Certifications, Summary)
- **Skills Extraction** – Detects 100+ technical and soft skills
- **Education Parsing** – Extracts degree, institution, year range, GPA
- **Named Entity Recognition** – Persons, organisations, locations via spaCy
- **Feature Extraction** – TF-IDF, Word2Vec, FastText, BERT, hand-crafted
- **Section Classifier** – Logistic Regression / LinearSVC on TF-IDF features
- **Evaluation** – Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
- **Interactive Demo** – Streamlit web application

## Project Structure

```
nlp-resume-parser/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Text extraction and cleaning
│   ├── feature_extraction.py       # TF-IDF, Word2Vec, FastText, BERT
│   ├── model.py                    # Parsing pipeline and classifiers
│   └── evaluation.py               # Metrics and plotting
├── notebooks/
│   └── resume_parsing_demo.ipynb   # Step-by-step notebook
├── data/
│   └── sample_resumes/             # Sample .txt resumes for testing
├── models/                         # Saved model artefacts
└── tests/
    ├── test_preprocessing.py
    ├── test_feature_extraction.py
    ├── test_model.py
    └── test_evaluation.py
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Install the spaCy English model (optional, improves NER):

```bash
python -m spacy download en_core_web_sm
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, upload a PDF or paste resume
text, and click **Parse Resume**.

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Open the Jupyter Notebook

```bash
jupyter notebook notebooks/resume_parsing_demo.ipynb
```

## Usage as a Library

```python
from src.model import ResumeParser

parser = ResumeParser(use_ner=True)

with open("data/sample_resumes/resume_software_engineer.txt") as f:
    text = f.read()

result = parser.parse(text)

print(result["contact"])
# {'email': 'john.doe@example.com', 'phone': '+1-555-123-4567', ...}

print(result["skills"]["technical"][:5])
# ['aws', 'docker', 'kubernetes', 'machine learning', 'python']

for entry in result["education"]:
    print(entry)
# {'degree': 'B.Tech', 'institution': '...', 'start_year': '2013', ...}
```

## Feature Extraction Examples

```python
from src.feature_extraction import TFIDFExtractor, HandcraftedFeatureExtractor

# TF-IDF
tfidf = TFIDFExtractor(max_features=5000)
X = tfidf.fit_transform(corpus)

# Hand-crafted features
hc = HandcraftedFeatureExtractor()
features = hc.extract(resume_text)
print(features["has_email"], features["word_count"])
```

## Evaluation

```python
from src.evaluation import compute_metrics, print_evaluation_summary

metrics = compute_metrics(y_true, y_pred)
print(metrics)
# {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.93, 'f1': 0.92}

print_evaluation_summary(y_true, y_pred, labels=section_labels)
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| NLP | spaCy, NLTK, Transformers (HuggingFace) |
| Embeddings | scikit-learn (TF-IDF), gensim (Word2Vec / FastText), PyTorch (BERT) |
| ML | scikit-learn |
| PDF Parsing | pdfplumber, PyPDF2 |
| Web App | Streamlit |
| Visualisation | matplotlib, seaborn, plotly |
| Testing | pytest |

## Dataset

Sample resumes are provided in `data/sample_resumes/` as plain-text files.
For model training, supplement with:

- [JANZZ Resume Dataset](https://github.com/nicholasmccullum/JANZZ-Dataset)
- CoNLL-2003 NER dataset (for NER training)
- Synthetically generated resumes using template scripts

## Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall fraction of correct predictions |
| Precision | TP / (TP + FP) – how many predicted positives are correct |
| Recall | TP / (TP + FN) – how many actual positives are captured |
| F1-Score | Harmonic mean of Precision and Recall |
| Confusion Matrix | Per-class breakdown of predictions |
| ROC-AUC | Area under the Receiver Operating Characteristic curve |

## License

This project is for academic use.
