import re

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize resume text for downstream NLP tasks."""
    if not text:
        return ""
    cleaned = text.replace("\u00a0", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip().lower()
    return cleaned


def extract_contact_entities(text: str) -> dict:
    """Extract contact-level entities from text."""
    if not text:
        return {"emails": [], "phones": []}
    return {
        "emails": sorted(set(EMAIL_PATTERN.findall(text))),
        "phones": sorted(set(PHONE_PATTERN.findall(text))),
    }
