FROM python:3.11.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    pip install \
      Flask==3.1.1 \
      flask-cors==6.0.1 \
      gunicorn==23.0.0 \
      requests==2.32.5 \
      Pillow==11.3.0 \
      pytesseract==0.3.13

COPY app.py .

CMD ["sh","-c","gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 1 --timeout 90"]
