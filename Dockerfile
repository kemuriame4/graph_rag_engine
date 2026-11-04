FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data

WORKDIR /app

COPY requirements.txt requirements-full.txt pyproject.toml ./
RUN pip install -r requirements.txt \
    && pip install "pymupdf>=1.24" "python-docx>=1.1" "qdrant-client>=1.9" \
       "elasticsearch>=8.13" "celery>=5.3" "redis>=5.0"

COPY app ./app
COPY graphrag ./graphrag
COPY samples ./samples
COPY scripts ./scripts

RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
