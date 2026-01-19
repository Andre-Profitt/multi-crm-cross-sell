FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs outputs models data config

ENV PYTHONPATH=/app:$PYTHONPATH

# Default command - can be overridden in docker-compose
CMD ["python", "main.py"]
