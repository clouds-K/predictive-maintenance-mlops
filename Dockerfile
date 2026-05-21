FROM python:3.10-slim

WORKDIR /app

COPY api/ ./api/

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn xgboost joblib numpy pydantic

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]