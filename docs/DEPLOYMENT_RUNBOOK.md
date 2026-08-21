# Phase 4 Deployment Runbook

## Prerequisites
- Docker and Docker Compose
- Trained models in `models/baseline/` and optional `models/advanced/`

## Local deployment
1. Copy `.env.example` to `.env` and set values.
2. Build and start services:
   ```bash
   docker compose up --build
   ```
3. Verify API health:
   ```bash
   curl http://localhost:8000/health
   ```
4. Open apps:
   - FastAPI docs: `http://localhost:8000/docs`
   - Streamlit app: `http://localhost:8501`

## Operational notes
- API includes optional bearer token authentication via `API_BEARER_TOKEN`.
- API includes configurable CORS and rate limiting.
- API health endpoint is used for container health checks.

## Troubleshooting
- Missing baseline model: run `python scripts/train.py`
- Missing advanced model: run `python scripts/train_advanced.py`
- Container restart loop: inspect logs with `docker compose logs -f`
