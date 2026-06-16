# Trading-AI / Agentic OS — base image for the API and the scheduler.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Kolkata \
    PYTHONPATH=/app

WORKDIR /app

# System deps (build tools for xgboost/scipy wheels are usually prebuilt; keep slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command is overridden per-service in docker-compose.
CMD ["python", "aos/scheduler.py", "status"]
