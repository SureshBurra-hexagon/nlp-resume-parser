from setuptools import find_packages, setup

setup(
    name="nlp-resume-parser",
    version="0.1.0",
    description="Phase 4 NLP resume parsing project with advanced parsing, API, web app, and deployment modules",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "streamlit>=1.30.0",
        "fastapi>=0.141.1",
        "uvicorn>=0.52.4",
    ],
)
