from src.preprocessing.text_cleaner import clean_text, extract_contact_entities


def test_clean_text_normalizes_case_and_spaces():
    text = "  DATA Scientist\nwith\tPython  "
    assert clean_text(text) == "data scientist with python"


def test_extract_contact_entities_returns_email_and_phone():
    text = "Reach me at test.user@example.com or +1 (555) 123-4567"
    entities = extract_contact_entities(text)
    assert "test.user@example.com" in entities["emails"]
    assert entities["phones"]
