FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    REDLINE_BACKEND=offline \
    REDLINE_DATA_DIR=/app/data \
    REDLINE_PLAYBOOK=/app/playbook/playbook.yaml
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY playbook ./playbook
COPY data ./data
RUN pip install . && useradd --create-home appuser

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "redline.api:app", "--host", "0.0.0.0", "--port", "8000"]
