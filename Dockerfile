# CPU-сборка по умолчанию (без GPU в образе). Для разработки без видеокарты достаточно этого.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/backend

# Совместимые версии с requirements.txt; без GPU — колёса с pytorch.org
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
