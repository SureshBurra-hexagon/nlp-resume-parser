"""
Streamlit web application for interactive resume parsing.

Run with:
    streamlit run app.py
"""

import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from src.preprocessing import extract_text_from_pdf, extract_text_from_txt
from src.model import ResumeParser

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NLP Resume Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_parser() -> ResumeParser:
    """Load and cache the ResumeParser instance."""
    return ResumeParser(use_ner=True)


def display_contact(contact: dict) -> None:
    cols = st.columns(3)
    fields = [
        ("📧 Email", contact.get("email")),
        ("📞 Phone", contact.get("phone")),
        ("🔗 LinkedIn", contact.get("linkedin")),
        ("🐙 GitHub", contact.get("github")),
        ("🌐 URL", contact.get("url")),
    ]
    for i, (label, value) in enumerate(fields):
        with cols[i % 3]:
            if value:
                st.metric(label=label, value=value)


def display_skills(skills: dict) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛠️ Technical Skills")
        if skills.get("technical"):
            for skill in skills["technical"]:
                st.markdown(f"- `{skill}`")
        else:
            st.info("No technical skills detected.")

    with col2:
        st.subheader("🤝 Soft Skills")
        if skills.get("soft"):
            for skill in skills["soft"]:
                st.markdown(f"- {skill}")
        else:
            st.info("No soft skills detected.")


def display_education(education: list) -> None:
    if not education:
        st.info("No education entries detected.")
        return
    for idx, entry in enumerate(education, 1):
        with st.expander(f"Education Entry {idx}", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Degree:** {entry.get('degree') or 'N/A'}")
                st.markdown(f"**Institution:** {entry.get('institution') or 'N/A'}")
            with cols[1]:
                start = entry.get("start_year") or "N/A"
                end = entry.get("end_year") or "N/A"
                st.markdown(f"**Duration:** {start} – {end}")
                st.markdown(f"**GPA:** {entry.get('gpa') or 'N/A'}")


def display_sections(sections: dict) -> None:
    skip = {"header"}
    for section, content in sections.items():
        if section in skip or not content.strip():
            continue
        with st.expander(section.replace("_", " ").title(), expanded=False):
            st.text(content[:2000])


def display_entities(entities: dict) -> None:
    if not entities:
        st.info("No named entities detected.")
        return
    labels_of_interest = ["PERSON", "ORG", "GPE", "DATE", "PRODUCT"]
    label_icons = {
        "PERSON": "👤",
        "ORG": "🏢",
        "GPE": "📍",
        "DATE": "📅",
        "PRODUCT": "📦",
    }
    for label in labels_of_interest:
        if label in entities and entities[label]:
            icon = label_icons.get(label, "🔹")
            st.markdown(f"**{icon} {label}:** {', '.join(entities[label][:10])}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📄 NLP Resume Parser")
    st.markdown(
        "Upload a PDF or paste plain text to extract structured information "
        "from a resume using Natural Language Processing."
    )

    # Sidebar
    st.sidebar.header("⚙️ Options")
    show_raw = st.sidebar.checkbox("Show raw extracted text", value=False)
    show_sections = st.sidebar.checkbox("Show identified sections", value=True)
    show_entities = st.sidebar.checkbox("Show named entities (NER)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tech Stack**\n"
        "- spaCy (NER)\n"
        "- NLTK (preprocessing)\n"
        "- scikit-learn (classification)\n"
        "- pdfplumber / PyPDF2 (PDF extraction)\n"
        "- Streamlit (UI)"
    )

    # Input
    tab1, tab2 = st.tabs(["📂 Upload File", "📝 Paste Text"])

    resume_text: str = ""

    with tab1:
        uploaded = st.file_uploader(
            "Upload a resume (PDF or TXT)", type=["pdf", "txt"]
        )
        if uploaded is not None:
            with st.spinner("Extracting text…"):
                try:
                    suffix = uploaded.name.rsplit(".", 1)[-1].lower()
                    if suffix == "pdf":
                        # Write to a temp file for pdfplumber
                        import tempfile  # noqa: PLC0415
                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False
                        ) as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name
                        resume_text = extract_text_from_pdf(tmp_path)
                        import os  # noqa: PLC0415
                        os.unlink(tmp_path)
                    else:
                        resume_text = uploaded.read().decode("utf-8", errors="replace")
                    st.success(f"Extracted {len(resume_text):,} characters.")
                except Exception as exc:
                    st.error(f"Failed to extract text: {exc}")

    with tab2:
        pasted = st.text_area(
            "Paste resume text here",
            height=300,
            placeholder="John Doe\njohn@example.com | +1 555-1234\n\nEducation\n…",
        )
        if pasted.strip():
            resume_text = pasted

    if not resume_text.strip():
        st.info("Please upload a file or paste resume text to begin.")
        return

    # Parse
    if st.button("🔍 Parse Resume", type="primary"):
        with st.spinner("Parsing resume…"):
            parser = get_parser()
            result = parser.parse(resume_text)

        st.success("Parsing complete!")
        st.markdown("---")

        # Contact info
        st.header("📇 Contact Information")
        display_contact(result["contact"])

        st.markdown("---")

        # Skills
        st.header("💡 Skills")
        display_skills(result["skills"])

        st.markdown("---")

        # Education
        st.header("🎓 Education")
        display_education(result["education"])

        # Named entities
        if show_entities:
            st.markdown("---")
            st.header("🏷️ Named Entities")
            display_entities(result["entities"])

        # Sections
        if show_sections:
            st.markdown("---")
            st.header("📑 Resume Sections")
            display_sections(result["sections"])

        # Raw text
        if show_raw:
            st.markdown("---")
            st.header("📃 Raw Text")
            st.text_area(
                "Extracted text",
                value=result["raw_text"][:5000],
                height=300,
                disabled=True,
            )

        # Download result as JSON
        import json  # noqa: PLC0415
        json_result = {
            k: v
            for k, v in result.items()
            if k != "raw_text"
        }
        st.download_button(
            label="⬇️ Download parsed result (JSON)",
            data=json.dumps(json_result, indent=2, default=str),
            file_name="parsed_resume.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
