FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs outputs models data config

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "main.py"]
