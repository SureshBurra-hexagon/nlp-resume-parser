"""
Model training for resume parsing tasks.

Provides:
- SectionClassifier  – multi-class classification of resume sections
- SkillsExtractor    – keyword-based skills extraction
- NERExtractor       – Named Entity Recognition via spaCy
- ResumeParser       – end-to-end parsing pipeline
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import LinearSVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from src.preprocessing import (
    PATTERNS,
    clean_text,
    extract_contact_info,
    identify_sections,
    preprocess_text,
)

# ---------------------------------------------------------------------------
# Skills database
# ---------------------------------------------------------------------------

TECHNICAL_SKILLS: List[str] = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "perl", "shell", "bash", "powershell",
    # Web / frontend
    "html", "css", "react", "angular", "vue", "node.js", "nodejs",
    "express", "django", "flask", "fastapi", "spring", "asp.net",
    # Data / ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "keras", "opencv",
    "bert", "gpt", "transformers", "huggingface",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "sqlite", "oracle", "dynamodb",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "ci/cd", "git", "github", "gitlab",
    "linux", "unix",
    # Misc tools
    "tableau", "power bi", "excel", "spark", "hadoop", "kafka",
    "airflow", "mlflow", "wandb", "jira", "confluence",
]

SOFT_SKILLS: List[str] = [
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "time management", "adaptability",
    "collaboration", "creativity", "analytical", "detail-oriented",
    "self-motivated", "fast learner", "multitasking",
    "presentation", "negotiation", "mentoring", "coaching",
]

ALL_SKILLS: List[str] = TECHNICAL_SKILLS + SOFT_SKILLS


# ---------------------------------------------------------------------------
# Section classifier
# ---------------------------------------------------------------------------


class SectionClassifier:
    """Multi-class classifier that labels chunks of text with resume sections.

    Uses a Logistic Regression (or LinearSVC) classifier on top of TF-IDF
    features.

    Args:
        model_type: ``"lr"`` for logistic regression, ``"svm"`` for LinearSVC.
        max_features: TF-IDF vocabulary size.
    """

    SECTION_LABELS = [
        "education", "experience", "skills", "projects",
        "certifications", "summary", "contact", "other",
    ]

    def __init__(
        self,
        model_type: str = "lr",
        max_features: int = 3000,
    ) -> None:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SectionClassifier.")

        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

        if model_type == "svm":
            clf = LinearSVC(max_iter=2000)
        else:
            clf = LogisticRegression(max_iter=1000, multi_class="auto")

        self.pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def fit(
        self,
        texts: List[str],
        labels: List[str],
    ) -> "SectionClassifier":
        """Train the classifier.

        Args:
            texts: List of text snippets (one per training example).
            labels: Corresponding section labels from ``SECTION_LABELS``.

        Returns:
            Self.
        """
        encoded = self.label_encoder.fit_transform(labels)
        self.pipeline.fit(texts, encoded)
        self._fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predict section labels for a list of text snippets.

        Args:
            texts: List of text strings.

        Returns:
            List of predicted section label strings.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        preds = self.pipeline.predict(texts)
        return list(self.label_encoder.inverse_transform(preds))

    def save(self, path: str) -> None:
        """Persist the pipeline and label encoder to disk."""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for saving models.")
        joblib.dump({"pipeline": self.pipeline, "encoder": self.label_encoder}, path)

    def load(self, path: str) -> "SectionClassifier":
        """Load a previously saved classifier."""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for loading models.")
        obj = joblib.load(path)
        self.pipeline = obj["pipeline"]
        self.label_encoder = obj["encoder"]
        self._fitted = True
        return self


# ---------------------------------------------------------------------------
# Skills extractor
# ---------------------------------------------------------------------------


class SkillsExtractor:
    """Extract technical and soft skills from resume text.

    Uses a keyword-matching approach with support for multi-word skills.

    Args:
        skills_list: Custom list of skills to look for.  Defaults to the
            built-in ``ALL_SKILLS`` list.
    """

    def __init__(self, skills_list: Optional[List[str]] = None) -> None:
        self.skills_list = [s.lower() for s in (skills_list or ALL_SKILLS)]
        # Sort by length descending so longer phrases are matched first
        self.skills_list.sort(key=len, reverse=True)
        self._technical_set = set(TECHNICAL_SKILLS)

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from a single text.

        Args:
            text: Raw or cleaned resume text.

        Returns:
            Dictionary with keys ``"technical"`` and ``"soft"`` mapping to
            sorted, deduplicated lists of matched skill strings.
        """
        text_lower = text.lower()
        found_technical: List[str] = []
        found_soft: List[str] = []

        for skill in self.skills_list:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                if skill in self._technical_set:
                    found_technical.append(skill)
                else:
                    found_soft.append(skill)

        return {
            "technical": sorted(set(found_technical)),
            "soft": sorted(set(found_soft)),
        }


# ---------------------------------------------------------------------------
# NER-based extractor
# ---------------------------------------------------------------------------


class NERExtractor:
    """Extract named entities using spaCy.

    Args:
        model_name: spaCy model name (default: ``"en_core_web_sm"``).
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is required for NERExtractor.")
        try:
            import spacy  # noqa: PLC0415
            self.nlp = spacy.load(model_name)
        except OSError:
            import spacy  # noqa: PLC0415
            self.nlp = spacy.blank("en")

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Run NER on text and group entities by label.

        Args:
            text: Input text.

        Returns:
            Dictionary mapping spaCy entity label (e.g. ``"ORG"``,
            ``"PERSON"``, ``"GPE"``) to a list of unique entity strings.
        """
        doc = self.nlp(text)
        entities: Dict[str, List[str]] = {}
        for ent in doc.ents:
            label = ent.label_
            entities.setdefault(label, [])
            if ent.text not in entities[label]:
                entities[label].append(ent.text)
        return entities


# ---------------------------------------------------------------------------
# Education extractor
# ---------------------------------------------------------------------------

_DEGREE_PATTERNS = re.compile(
    r"\b(b\.?s\.?|b\.?e\.?|b\.?tech|b\.?sc|bachelor(?:\'s)?|"
    r"m\.?s\.?|m\.?e\.?|m\.?tech|m\.?sc|master(?:\'s)?|"
    r"ph\.?d\.?|doctorate|associate(?:\'s)?|diploma|"
    r"b\.?a\.?|m\.?b\.?a\.?)\b",
    re.IGNORECASE,
)

_YEAR_RANGE_PATTERN = re.compile(
    r"((?:19|20)\d{2})\s*[-–to]+\s*((?:19|20)\d{2}|present|current)",
    re.IGNORECASE,
)


def extract_education(text: str) -> List[Dict[str, Optional[str]]]:
    """Rule-based education extraction from the education section text.

    Args:
        text: Text from the education section of a resume.

    Returns:
        List of dictionaries with keys ``degree``, ``institution``,
        ``start_year``, ``end_year``, ``gpa``.
    """
    results: List[Dict[str, Optional[str]]] = []

    # Each block separated by blank line or long whitespace is likely one entry
    blocks = re.split(r"\n{2,}", text) if "\n\n" in text else [text]

    for block in blocks:
        if not block.strip():
            continue

        degree_match = _DEGREE_PATTERNS.search(block)
        year_match = _YEAR_RANGE_PATTERN.search(block)
        gpa_match = PATTERNS["gpa"].search(block)

        if degree_match or year_match:
            entry: Dict[str, Optional[str]] = {
                "degree": degree_match.group().strip() if degree_match else None,
                "institution": _extract_institution(block),
                "start_year": year_match.group(1) if year_match else None,
                "end_year": year_match.group(2) if year_match else None,
                "gpa": gpa_match.group(1).strip() if gpa_match else None,
            }
            results.append(entry)

    return results


def _extract_institution(text: str) -> Optional[str]:
    """Heuristic: first line of the block is likely the institution name."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) > 3:
            return stripped
    return None


# ---------------------------------------------------------------------------
# End-to-end parser
# ---------------------------------------------------------------------------


class ResumeParser:
    """End-to-end resume parsing pipeline.

    Combines section identification, contact extraction, skills extraction,
    education parsing, and NER into a single ``parse()`` call.

    Args:
        use_ner: Whether to run spaCy NER (requires spaCy + model).
    """

    def __init__(self, use_ner: bool = True) -> None:
        self.skills_extractor = SkillsExtractor()
        self.ner_extractor: Optional[NERExtractor] = None
        if use_ner and SPACY_AVAILABLE:
            try:
                self.ner_extractor = NERExtractor()
            except Exception:
                self.ner_extractor = None

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse a resume and return structured information.

        Args:
            text: Full resume text (extracted from PDF or plain text).

        Returns:
            Dictionary with keys:
            - ``raw_text`` – cleaned input text
            - ``sections``  – dict mapping section name → raw section text
            - ``contact``   – extracted contact details
            - ``skills``    – ``{"technical": [...], "soft": [...]}``
            - ``education`` – list of education entries
            - ``entities``  – NER entities (empty dict if NER disabled)
        """
        text = clean_text(text)
        sections = identify_sections(text)
        contact = extract_contact_info(text)

        # Skills: prioritise the skills section, fall back to full text
        skills_text = sections.get("skills", text)
        skills = self.skills_extractor.extract(skills_text + "\n" + text)

        # Education
        education_text = sections.get("education", "")
        education = extract_education(education_text)

        # NER
        entities: Dict[str, List[str]] = {}
        if self.ner_extractor is not None:
            try:
                entities = self.ner_extractor.extract(text[:5000])
            except Exception:
                entities = {}

        return {
            "raw_text": text,
            "sections": sections,
            "contact": contact,
            "skills": skills,
            "education": education,
            "entities": entities,
        }
