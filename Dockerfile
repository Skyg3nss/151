FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# System Tesseract is dramatically lighter in RAM than EasyOCR/PyTorch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the upstream project WITHOUT dependencies so pip cannot pull
# EasyOCR/PyTorch. Then install only the lightweight pieces needed by the
# bundled reference + WordClassifier.
RUN python -m pip install --upgrade pip && \
    pip install --no-deps pokemon-card-recognizer==0.0.1.3.8.7 && \
    pip install --no-deps ocr-ops==0.0.0.4.3.1 && \
    pip install algo-ops==0.0.1.7.1 && \
    pip install \
      numpy==1.26.4 \
      pandas==2.2.3 \
      pokemontcgsdk \
      bidict \
      ordered-set \
      requests \
      Pillow==9.5.0 \
      tqdm \
      pytesseract==0.3.13 \
      Flask==3.1.1 \
      flask-cors==6.0.1 \
      gunicorn==23.0.0

COPY app.py .

# One worker only: Render Free has 512 MiB.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 1 --timeout 120 --max-requests 40 --max-requests-jitter 5"]
