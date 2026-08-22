from __future__ import annotations

import re

from src.preprocessing.text_cleaner import clean_text

SECTION_ALIASES = {
    "summary": {"summary", "profile", "objective", "about"},
    "experience": {"experience", "work experience", "employment", "professional experience"},
    "education": {"education", "academic background", "academics"},
    "skills": {"skills", "technical skills", "core skills", "competencies"},
    "projects": {"projects", "project experience"},
    "certifications": {"certifications", "licenses", "credentials"},
}

DEGREE_PATTERNS = {
    "bachelor": [
        r"\bbachelor(?:'s)?(?: of [a-z& ]+)?\b",
        r"\bb\.?\s?(?:sc|tech|e|a)\b",
        r"\bbs\b",
    ],
    "master": [
        r"\bmaster(?:'s)?(?: of [a-z& ]+)?\b",
        r"\bm\.?\s?(?:sc|tech|e|a)\b",
        r"\bms\b",
    ],
    "mba": [r"\bmba\b", r"\bmaster of business administration\b"],
    "phd": [r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctoral\b"],
}

CERTIFICATION_PATTERNS = {
    "aws certified": [r"\baws certified(?: [a-z ]+)?\b"],
    "azure certified": [r"\bazure certified(?: [a-z ]+)?\b"],
    "gcp certified": [r"\bgcp certified(?: [a-z ]+)?\b", r"\bgoogle cloud certified(?: [a-z ]+)?\b"],
    "pmp": [r"\bpmp\b", r"\bproject management professional\b"],
    "scrum master": [r"\bscrum master\b", r"\bcertified scrum master\b", r"\bcsm\b"],
}

YEARS_PATTERN = re.compile(r"\b(\d{1,2})\+?\s+years?\b", re.IGNORECASE)


def _match_section_name(value: str) -> str | None:
    normalized = clean_text(value).rstrip(":")
    for canonical, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def extract_sections(text: str) -> dict[str, str]:
    if not text:
        return {}

    sections: dict[str, list[str]] = {}
    current_section = "summary"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if ":" in line:
            maybe_header, remainder = line.split(":", 1)
            matched_section = _match_section_name(maybe_header)
            if matched_section:
                current_section = matched_section
                sections.setdefault(current_section, [])
                remainder = remainder.strip()
                if remainder:
                    sections[current_section].append(clean_text(remainder))
                continue

        matched_section = _match_section_name(line)
        if matched_section:
            current_section = matched_section
            sections.setdefault(current_section, [])
            continue

        sections.setdefault(current_section, []).append(clean_text(line))

    return {
        section: " ".join(part for part in content if part).strip()
        for section, content in sections.items()
        if any(content)
    }


def _extract_canonical_matches(text: str, patterns: dict[str, list[str]]) -> list[str]:
    normalized = clean_text(text)
    matches: list[str] = []
    for canonical, regexes in patterns.items():
        if any(re.search(regex, normalized, re.IGNORECASE) for regex in regexes):
            matches.append(canonical)
    return sorted(matches)


def extract_education_entities(text: str) -> list[str]:
    return _extract_canonical_matches(text, DEGREE_PATTERNS)


def extract_certifications(text: str) -> list[str]:
    return _extract_canonical_matches(text, CERTIFICATION_PATTERNS)


def estimate_experience_years(text: str) -> int | None:
    years = [int(match.group(1)) for match in YEARS_PATTERN.finditer(text)]
    if not years:
        return None
    return max(years)
