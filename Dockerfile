FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files for dependency installation
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Copy remaining files
COPY . .

RUN mkdir -p logs outputs models data config

ENV PYTHONPATH=/app:$PYTHONPATH

# Default command - can be overridden in docker-compose
CMD ["python", "main.py"]
