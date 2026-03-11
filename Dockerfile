FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY memoria/ memoria/

RUN pip install --no-cache-dir .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "memoria.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
