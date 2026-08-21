from src.augmentation.resume_augmenter import augment_resume_text, augment_training_data


def test_augment_resume_text_generates_deterministic_variants():
    variants = augment_resume_text("Data scientist with machine learning and React experience", max_variants=3)

    assert len(variants) == 3
    assert any("ml" in variant for variant in variants)
    assert any("react.js" in variant for variant in variants)


def test_augment_training_data_preserves_labels():
    texts, labels = augment_training_data(
        ["Python NLP engineer", "Frontend developer with React"],
        ["data_science", "frontend"],
        max_variants_per_text=1,
    )

    assert len(texts) == len(labels) == 4
    assert labels.count("data_science") == 2
    assert labels.count("frontend") == 2
