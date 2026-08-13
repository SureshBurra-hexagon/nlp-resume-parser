"""
Text preprocessing module for resume parsing.

Handles:
- PDF/text extraction
- Tokenization and sentence segmentation
- Lowercasing, punctuation removal
- Stopword removal
- Lemmatization
- Section identification
"""

import re
import string
from typing import Dict, List, Optional, Tuple

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# Section header keywords used to identify resume sections
SECTION_HEADERS: Dict[str, List[str]] = {
    "education": [
        "education", "academic background", "academic qualifications",
        "qualifications", "degrees", "university", "college",
    ],
    "experience": [
        "experience", "work experience", "employment history",
        "professional experience", "career history", "work history",
        "internship", "internships",
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "technologies", "tools", "expertise", "proficiencies",
    ],
    "projects": [
        "projects", "personal projects", "academic projects",
        "notable projects", "portfolio",
    ],
    "certifications": [
        "certifications", "certificates", "licenses", "accreditations",
        "professional certifications",
    ],
    "summary": [
        "summary", "objective", "profile", "about me", "professional summary",
        "career objective",
    ],
    "contact": [
        "contact", "contact information", "personal information",
        "personal details",
    ],
}

# Compiled regex patterns for common resume fields.
# Note: email is extracted via a two-step whitespace-split + validation
# approach to eliminate polynomial ReDoS risk on adversarial input.
_EMAIL_VALIDATION = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9_\-]*(?:\.[a-zA-Z0-9_\-]+)*"
    r"@[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)+$",
    re.IGNORECASE,
)

PATTERNS = {
    "phone": re.compile(
        r"\+?\d{1,3}[\s\-.]?\(?\d{1,4}\)?[\s\-.]?\d{1,4}[\s\-.]?\d{1,9}",
        re.IGNORECASE
    ),
    "url": re.compile(
        r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE
    ),
    "linkedin": re.compile(
        r"linkedin\.com/in/[a-zA-Z0-9\-_%]+", re.IGNORECASE
    ),
    "github": re.compile(
        r"github\.com/[a-zA-Z0-9\-_%]+", re.IGNORECASE
    ),
    "year": re.compile(r"\b(19|20)\d{2}\b"),
    "gpa": re.compile(
        r"(?:GPA|CGPA|Grade)[:\s]*([0-4]\.\d{1,2}(?:\s*/\s*[0-4](?:\.\d)?)?)",
        re.IGNORECASE,
    ),
}


def _extract_email(text: str) -> Optional[str]:
    """Extract the first email address from *text* without ReDoS risk.

    Uses a whitespace-split approach: each token that contains ``@`` is
    validated individually with a bounded regex, eliminating the nested
    quantifier backtracking that causes polynomial ReDoS.

    Args:
        text: Input text.

    Returns:
        First valid email string found, or ``None``.
    """
    for token in text.split():
        # Strip common surrounding punctuation
        candidate = token.strip("(),;:\"'<>[]")
        if "@" not in candidate:
            continue
        # Limit candidate length to avoid edge cases
        if len(candidate) > 254:
            continue
        if _EMAIL_VALIDATION.match(candidate):
            return candidate
    return None


def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from a PDF file.

    Attempts pdfplumber first for better layout handling, falls back to PyPDF2.

    Args:
        file_path: Absolute or relative path to a PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If no PDF library is available.
    """
    if not PDFPLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
        raise RuntimeError(
            "Neither pdfplumber nor PyPDF2 is installed. "
            "Run: pip install pdfplumber PyPDF2"
        )

    text_parts: List[str] = []

    if PDFPLUMBER_AVAILABLE:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    # Fallback to PyPDF2
    with open(file_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_txt(file_path: str) -> str:
    """Read plain text from a .txt file.

    Args:
        file_path: Path to the text file.

    Returns:
        File contents as a string.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def clean_text(text: str) -> str:
    """Apply basic text cleaning.

    Steps:
    1. Collapse excessive whitespace / blank lines.
    2. Remove non-printable / control characters.
    3. Normalize unicode dashes and quotes.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    # Normalize unicode dashes and apostrophes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Remove non-printable characters except newlines and tabs
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e\xa0-\xff]", " ", text)

    # Collapse multiple blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text into words.

    Uses NLTK word_tokenize when available, falls back to a simple regex split.

    Args:
        text: Input text.

    Returns:
        List of word tokens.
    """
    if NLTK_AVAILABLE:
        try:
            _ensure_nltk_resources()
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"\b\w+\b", text)


def sentence_tokenize(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    if NLTK_AVAILABLE:
        try:
            _ensure_nltk_resources()
            return sent_tokenize(text)
        except Exception:
            pass
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def remove_stopwords(tokens: List[str], language: str = "english") -> List[str]:
    """Remove stopwords from a token list.

    Args:
        tokens: List of word tokens.
        language: Language for NLTK stopwords (default: ``"english"``).

    Returns:
        Filtered token list with stopwords removed.
    """
    if NLTK_AVAILABLE:
        try:
            _ensure_nltk_resources()
            stop_words = set(stopwords.words(language))
            return [t for t in tokens if t.lower() not in stop_words]
        except Exception:
            pass
    # Minimal English stopwords fallback
    _minimal = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "their", "this", "that", "these", "those",
    }
    return [t for t in tokens if t.lower() not in _minimal]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using spaCy when available.

    Falls back to lowercasing when spaCy is unavailable.

    Args:
        tokens: List of word tokens.

    Returns:
        List of lemmatized tokens.
    """
    if SPACY_AVAILABLE:
        try:
            nlp = _get_spacy_model()
            doc = nlp(" ".join(tokens))
            return [token.lemma_ for token in doc]
        except Exception:
            pass
    return [t.lower() for t in tokens]


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_stops: bool = True,
    do_lemmatize: bool = True,
) -> List[str]:
    """Full preprocessing pipeline.

    Applies cleaning → tokenization → optional lowercasing →
    optional punctuation removal → optional stopword removal →
    optional lemmatization.

    Args:
        text: Raw resume text.
        lowercase: Convert tokens to lowercase.
        remove_punct: Strip punctuation tokens.
        remove_stops: Remove stopwords.
        do_lemmatize: Apply lemmatization.

    Returns:
        Processed list of tokens.
    """
    text = clean_text(text)
    tokens = tokenize(text)

    if lowercase:
        tokens = [t.lower() for t in tokens]

    if remove_punct:
        punct = set(string.punctuation)
        tokens = [t for t in tokens if t not in punct]

    if remove_stops:
        tokens = remove_stopwords(tokens)

    if do_lemmatize:
        tokens = lemmatize(tokens)

    return [t for t in tokens if t.strip()]


def identify_sections(text: str) -> Dict[str, str]:
    """Split a resume into named sections.

    Scans each line for known section-header keywords.  Text between
    consecutive section headers is assigned to the preceding header.

    Args:
        text: Full resume text (may contain newlines).

    Returns:
        Dictionary mapping section name → section text.
    """
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {}
    current_section = "header"
    sections[current_section] = []

    for line in lines:
        stripped = line.strip().lower()
        matched_section: Optional[str] = None

        for section, keywords in SECTION_HEADERS.items():
            for kw in keywords:
                # Match keyword as whole line or with trailing colon/space
                if stripped == kw or re.match(
                    r"^" + re.escape(kw) + r"[\s:]*$", stripped
                ):
                    matched_section = section
                    break
            if matched_section:
                break

        if matched_section:
            current_section = matched_section
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """Extract contact information from resume text using regex patterns.

    Args:
        text: Raw or cleaned resume text.

    Returns:
        Dictionary with keys ``email``, ``phone``, ``linkedin``,
        ``github``, ``url``.  Values are ``None`` when not found.
    """
    result: Dict[str, Optional[str]] = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "url": None,
    }

    result["email"] = _extract_email(text)

    phone_match = PATTERNS["phone"].search(text)
    if phone_match:
        result["phone"] = phone_match.group().strip()

    linkedin_match = PATTERNS["linkedin"].search(text)
    if linkedin_match:
        result["linkedin"] = linkedin_match.group()

    github_match = PATTERNS["github"].search(text)
    if github_match:
        result["github"] = github_match.group()

    url_match = PATTERNS["url"].search(text)
    if url_match:
        result["url"] = url_match.group()

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_nlp_model = None


def _get_spacy_model():
    """Load and cache the spaCy language model."""
    global _nlp_model
    if _nlp_model is None:
        import spacy  # noqa: PLC0415

        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            _nlp_model = spacy.blank("en")
    return _nlp_model


_nltk_resources_downloaded = False


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present."""
    global _nltk_resources_downloaded
    if _nltk_resources_downloaded:
        return
    import nltk  # noqa: PLC0415

    resource_paths = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    }
    for resource, find_path in resource_paths.items():
        try:
            nltk.data.find(find_path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    _nltk_resources_downloaded = True
