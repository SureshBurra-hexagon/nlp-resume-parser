from __future__ import annotations

from functools import lru_cache

from src.utils.resume_parser import parse_resume


@lru_cache(maxsize=256)
def _cached_parsed_resume(text: str) -> dict:
    return parse_resume(text)


def cached_parse_resume(text: str) -> dict:
    parsed = _cached_parsed_resume(text)
    return {
        "normalized_text": parsed["normalized_text"],
        "emails": list(parsed["emails"]),
        "phones": list(parsed["phones"]),
        "skills": list(parsed["skills"]),
        "sections": dict(parsed["sections"]),
        "education": list(parsed["education"]),
        "certifications": list(parsed["certifications"]),
        "experience_years": parsed["experience_years"],
    }


class ResumeBatchProcessor:
    def parse(self, text: str) -> dict:
        return cached_parse_resume(text)

    def parse_batch(self, texts: list[str]) -> list[dict]:
        return [self.parse(text) for text in texts]

    def normalized_batch(self, texts: list[str]) -> list[str]:
        return [parsed["normalized_text"] for parsed in self.parse_batch(texts)]
