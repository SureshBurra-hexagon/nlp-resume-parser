.PHONY: test train evaluate train-advanced evaluate-advanced streamlit api

test:
	pytest -q

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

train-advanced:
	python scripts/train_advanced.py

evaluate-advanced:
	python scripts/evaluate_advanced.py

streamlit:
	streamlit run streamlit_app/app.py

api:
	uvicorn fastapi_app.main:app --reload
