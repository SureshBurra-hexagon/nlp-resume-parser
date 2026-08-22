from __future__ import annotations

from src.preprocessing.text_cleaner import clean_text

PHRASE_REPLACEMENTS = [
    ("machine learning", "ml"),
    ("frontend", "front-end"),
    ("devops", "platform engineering"),
    ("business analyst", "analytics specialist"),
    ("data scientist", "ml practitioner"),
    ("react", "react.js"),
    ("ci/cd", "continuous delivery"),
]

APPEND_TEMPLATES = [
    "delivered measurable project outcomes",
    "collaborated with cross-functional teams",
    "optimized production workflows",
]


def augment_resume_text(text: str, max_variants: int = 2) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []

    variants: list[str] = []
    seen = {normalized}

    for original, replacement in PHRASE_REPLACEMENTS:
        if len(variants) >= max_variants:
            break
        if original in normalized:
            candidate = normalized.replace(original, replacement, 1)
            if candidate not in seen:
                variants.append(candidate)
                seen.add(candidate)

    for template in APPEND_TEMPLATES:
        if len(variants) >= max_variants:
            break
        candidate = f"{normalized} {template}"
        if candidate not in seen:
            variants.append(candidate)
            seen.add(candidate)

    return variants


def augment_training_data(
    texts: list[str],
    labels: list[str],
    max_variants_per_text: int = 2,
) -> tuple[list[str], list[str]]:
    augmented_texts: list[str] = []
    augmented_labels: list[str] = []

    for text, label in zip(texts, labels, strict=True):
        normalized = clean_text(text)
        augmented_texts.append(normalized)
        augmented_labels.append(label)

        for variant in augment_resume_text(normalized, max_variants=max_variants_per_text):
            augmented_texts.append(variant)
            augmented_labels.append(label)

    return augmented_texts, augmented_labels
