.PHONY: test train evaluate streamlit api

test:
	pytest -q

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

streamlit:
	streamlit run streamlit_app/app.py

api:
	uvicorn fastapi_app.main:app --reload
