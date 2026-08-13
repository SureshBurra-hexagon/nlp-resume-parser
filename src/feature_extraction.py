"""
Feature extraction and embeddings for resume parsing.

Implements:
- TF-IDF vectorisation
- Word2Vec embeddings (via gensim)
- FastText embeddings (via gensim)
- BERT contextual embeddings (via transformers)
- Hand-crafted regex features
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from gensim.models import FastText, Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


class TFIDFExtractor:
    """Wrap sklearn's TfidfVectorizer with a simple train/transform API.

    Args:
        max_features: Maximum number of vocabulary terms.
        ngram_range: Lower and upper n-gram boundaries.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        **kwargs: Any,
    ) -> None:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TFIDFExtractor.")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            **kwargs,
        )
        self._fitted = False

    def fit(self, corpus: List[str]) -> "TFIDFExtractor":
        """Fit the vectorizer on a list of documents.

        Args:
            corpus: List of text strings (one per document).

        Returns:
            Self (for chaining).
        """
        self.vectorizer.fit(corpus)
        self._fitted = True
        return self

    def transform(self, corpus: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF feature matrix.

        Args:
            corpus: List of text strings.

        Returns:
            Dense numpy array of shape ``(n_docs, max_features)``.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return self.vectorizer.transform(corpus).toarray()

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        """Fit then transform in one step.

        Args:
            corpus: List of text strings.

        Returns:
            Dense numpy array.
        """
        self._fitted = True
        return self.vectorizer.fit_transform(corpus).toarray()

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Return the fitted vocabulary mapping."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.vectorizer.vocabulary_


# ---------------------------------------------------------------------------
# Word2Vec
# ---------------------------------------------------------------------------


class Word2VecExtractor:
    """Train or load a Word2Vec model and produce document embeddings.

    Document embeddings are computed as the mean of token embeddings,
    ignoring tokens not present in the vocabulary.

    Args:
        vector_size: Dimensionality of word vectors.
        window: Context window size.
        min_count: Minimum token frequency.
        workers: Number of training threads.
        sg: Training algorithm: 1 for skip-gram, 0 for CBOW.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 1,
    ) -> None:
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for Word2VecExtractor.")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model: Optional[Word2Vec] = None

    def fit(self, tokenized_corpus: List[List[str]]) -> "Word2VecExtractor":
        """Train Word2Vec on a tokenised corpus.

        Args:
            tokenized_corpus: List of token lists (one per document).

        Returns:
            Self.
        """
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
        )
        return self

    def transform(self, tokenized_corpus: List[List[str]]) -> np.ndarray:
        """Compute mean-pooled document embeddings.

        Args:
            tokenized_corpus: List of token lists.

        Returns:
            Array of shape ``(n_docs, vector_size)``.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")
        embeddings = []
        for tokens in tokenized_corpus:
            vecs = [
                self.model.wv[t]
                for t in tokens
                if t in self.model.wv
            ]
            if vecs:
                embeddings.append(np.mean(vecs, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)

    def save(self, path: str) -> None:
        """Persist the model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)

    def load(self, path: str) -> "Word2VecExtractor":
        """Load a previously saved model."""
        self.model = Word2Vec.load(path)
        self.vector_size = self.model.vector_size
        return self


# ---------------------------------------------------------------------------
# FastText
# ---------------------------------------------------------------------------


class FastTextExtractor:
    """Train or load a FastText model and produce document embeddings.

    Unlike Word2Vec, FastText can handle out-of-vocabulary words via
    subword information.

    Args:
        vector_size: Dimensionality of word vectors.
        window: Context window size.
        min_count: Minimum token frequency.
        workers: Number of training threads.
        sg: 1 for skip-gram, 0 for CBOW.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 1,
    ) -> None:
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for FastTextExtractor.")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model: Optional[FastText] = None

    def fit(self, tokenized_corpus: List[List[str]]) -> "FastTextExtractor":
        """Train FastText on a tokenised corpus.

        Args:
            tokenized_corpus: List of token lists.

        Returns:
            Self.
        """
        self.model = FastText(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
        )
        return self

    def transform(self, tokenized_corpus: List[List[str]]) -> np.ndarray:
        """Compute mean-pooled document embeddings.

        Args:
            tokenized_corpus: List of token lists.

        Returns:
            Array of shape ``(n_docs, vector_size)``.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")
        embeddings = []
        for tokens in tokenized_corpus:
            if tokens:
                vecs = [self.model.wv[t] for t in tokens]
                embeddings.append(np.mean(vecs, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)

    def save(self, path: str) -> None:
        """Persist the model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)

    def load(self, path: str) -> "FastTextExtractor":
        """Load a previously saved model."""
        self.model = FastText.load(path)
        self.vector_size = self.model.vector_size
        return self


# ---------------------------------------------------------------------------
# BERT
# ---------------------------------------------------------------------------


class BERTExtractor:
    """Produce CLS-token embeddings using a pre-trained BERT model.

    Args:
        model_name: HuggingFace model identifier
            (default: ``"bert-base-uncased"``).
        device: Torch device string.  ``None`` auto-selects CUDA / CPU.
        max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for BERTExtractor."
            )
        self.model_name = model_name
        self.max_length = max_length
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts to CLS-token embeddings.

        Args:
            texts: List of text strings.

        Returns:
            Array of shape ``(n_texts, hidden_size)``.
        """
        all_embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # CLS token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                all_embeddings.append(cls_embedding.cpu().numpy())
        return np.array(all_embeddings)


# ---------------------------------------------------------------------------
# Hand-crafted features
# ---------------------------------------------------------------------------


class HandcraftedFeatureExtractor:
    """Extract hand-crafted boolean / count features from resume text.

    Feature list
    ------------
    ``has_email``, ``has_phone``, ``has_linkedin``, ``has_github``,
    ``has_url``, ``num_years_mentioned``, ``has_gpa``,
    ``word_count``, ``char_count``, ``sentence_count``,
    ``num_bullet_points``, ``has_education_section``,
    ``has_experience_section``, ``has_skills_section``,
    ``has_projects_section``, ``has_certifications_section``
    """

    _EMAIL_VALIDATION = re.compile(
        r"^[a-zA-Z0-9][a-zA-Z0-9_\-]*(?:\.[a-zA-Z0-9_\-]+)*"
        r"@[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)+$",
        re.IGNORECASE,
    )
    _PHONE = re.compile(r"\+?\d{1,3}[\s\-.]?\(?\d{1,4}\)?[\s\-.]?\d{1,4}[\s\-.]?\d{1,9}")
    _LINKEDIN = re.compile(r"linkedin\.com/in/", re.IGNORECASE)
    _GITHUB = re.compile(r"github\.com/", re.IGNORECASE)
    _URL = re.compile(r"https?://|www\.", re.IGNORECASE)
    _YEAR = re.compile(r"\b(19|20)\d{2}\b")
    _GPA = re.compile(r"(?:GPA|CGPA)[:\s]*[0-4]\.\d", re.IGNORECASE)
    _BULLET = re.compile(r"^\s*[•\-\*\u2022]", re.MULTILINE)

    _SECTION_PATTERNS = {
        "education": re.compile(r"\beducation\b", re.IGNORECASE),
        "experience": re.compile(r"\bexperience\b", re.IGNORECASE),
        "skills": re.compile(r"\bskills\b", re.IGNORECASE),
        "projects": re.compile(r"\bprojects\b", re.IGNORECASE),
        "certifications": re.compile(r"\bcertification", re.IGNORECASE),
    }

    def _has_email(self, text: str) -> bool:
        """Check for an email address using a safe token-validation approach."""
        for token in text.split():
            candidate = token.strip("(),;:\"'<>[]")
            if "@" not in candidate or len(candidate) > 254:
                continue
            if self._EMAIL_VALIDATION.match(candidate):
                return True
        return False

    def extract(self, text: str) -> Dict[str, Union[int, float, bool]]:
        """Extract hand-crafted features from a single text.

        Args:
            text: Resume text.

        Returns:
            Dictionary of feature name → numeric or boolean value.
        """
        sentences = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
        features: Dict[str, Union[int, float, bool]] = {
            "has_email": self._has_email(text),
            "has_phone": bool(self._PHONE.search(text)),
            "has_linkedin": bool(self._LINKEDIN.search(text)),
            "has_github": bool(self._GITHUB.search(text)),
            "has_url": bool(self._URL.search(text)),
            "num_years_mentioned": len(self._YEAR.findall(text)),
            "has_gpa": bool(self._GPA.search(text)),
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentence_count": len(sentences),
            "num_bullet_points": len(self._BULLET.findall(text)),
        }
        for section, pattern in self._SECTION_PATTERNS.items():
            features[f"has_{section}_section"] = bool(pattern.search(text))
        return features

    def extract_batch(
        self, texts: List[str]
    ) -> List[Dict[str, Union[int, float, bool]]]:
        """Extract features from a list of texts.

        Args:
            texts: List of resume text strings.

        Returns:
            List of feature dictionaries.
        """
        return [self.extract(t) for t in texts]

    def to_matrix(self, texts: List[str]) -> np.ndarray:
        """Return hand-crafted features as a numeric matrix.

        Args:
            texts: List of resume text strings.

        Returns:
            Array of shape ``(n_texts, n_features)`` with float values.
        """
        batch = self.extract_batch(texts)
        if not batch:
            return np.array([])
        keys = list(batch[0].keys())
        rows = [[float(d[k]) for k in keys] for d in batch]
        return np.array(rows, dtype=float)

    @property
    def feature_names(self) -> List[str]:
        """Return ordered feature name list."""
        dummy = self.extract("dummy text")
        return list(dummy.keys())
