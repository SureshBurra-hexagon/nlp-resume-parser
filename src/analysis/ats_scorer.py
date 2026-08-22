from __future__ import annotations

import re
from collections import Counter

from src.preprocessing.text_cleaner import clean_text

_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "for", "with",
        "on", "at", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "that",
        "this", "these", "those", "it", "its", "we", "our", "you", "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+/.\-]*")


def _tokenize(text: str) -> list[str]:
    normalized = clean_text(text)
    return [t for t in _TOKEN_RE.findall(normalized) if t not in _STOP_WORDS and len(t) > 1]


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def extract_jd_keywords(job_description: str, top_n: int = 30) -> list[str]:
    """Return the most frequent meaningful unigrams and bigrams from a job description."""
    tokens = _tokenize(job_description)
    candidates = tokens + _bigrams(tokens)
    counter = Counter(candidates)
    return [kw for kw, _ in counter.most_common(top_n)]


def compute_ats_score(resume_text: str, job_description: str, top_n: int = 30) -> dict:
    """Score how well a resume matches a job description.

    Returns a dict with:
      - ``score``: float in [0, 1] — fraction of top JD keywords found in the resume
      - ``matched_keywords``: sorted list of matched keywords
      - ``missing_keywords``: sorted list of unmatched keywords
      - ``total_keywords``: number of JD keywords evaluated
    """
    jd_keywords = extract_jd_keywords(job_description, top_n=top_n)
    if not jd_keywords:
        return {
            "score": 0.0,
            "matched_keywords": [],
            "missing_keywords": [],
            "total_keywords": 0,
        }

    resume_normalized = clean_text(resume_text)
    resume_tokens = set(_tokenize(resume_text))
    resume_bigrams = set(_bigrams(list(resume_tokens)))
    resume_vocab = resume_tokens | resume_bigrams | {resume_normalized}

    matched = sorted({kw for kw in jd_keywords if kw in resume_vocab or kw in resume_normalized})
    missing = sorted(set(jd_keywords) - set(matched))

    return {
        "score": len(matched) / len(jd_keywords),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "total_keywords": len(jd_keywords),
    }
