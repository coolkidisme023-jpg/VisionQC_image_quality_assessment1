# VisionQC backend image.
#
# Build context MUST be the repository root (not backend/), because this
# image also needs the trained model artifacts from ../models. From the
# repo root:
#     docker build -f backend/Dockerfile -t visionqc-backend .
# (docker-compose.yml already sets this up correctly.)
#
# Container layout mirrors the repo layout so app/config.py's path
# resolution (REPO_ROOT = <backend dir>.parent) works unchanged:
#     /app/backend/...   (application code)
#     /app/models/...    (trained model + metrics, copied at build time)
#     /app/data/...      (SQLite database - bind-mounted volume in compose)

FROM python:3.11-slim

# Runtime OpenCV needs a couple of shared libraries not present in the
# slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY models/ models/

RUN mkdir -p /app/data

WORKDIR /app/backend

ENV PYTHONUNBUFFERED=1 \
    VISIONQC_HOST=0.0.0.0 \
    VISIONQC_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# gunicorn (production WSGI server) serves the Flask `app` object exposed
# by app/main.py. 2 workers is a reasonable default for a small deployment.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60", "app.main:app"]
