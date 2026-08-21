.PHONY: test train evaluate train-advanced evaluate-advanced train-transformer evaluate-transformer parse-file streamlit api

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

train-transformer:
	python scripts/train_transformer.py

evaluate-transformer:
	python scripts/evaluate_transformer.py

parse-file:
	@test -n "$(FILE)" || (echo "Usage: make parse-file FILE=/absolute/path/to/resume.(txt|html|docx|pdf)" && exit 1)
	python scripts/parse_file.py --file "$(FILE)"

streamlit:
	streamlit run streamlit_app/app.py

api:
	uvicorn fastapi_app.main:app --reload
