FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

WORKDIR /app

# Установим переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Установим зависимости для сборки fasttext
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    python3-dev \
    python3-pip \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Скопируем зависимости
COPY requirements.txt .
COPY data/processed/faiss_index.index /app/data/processed/faiss_index.index
COPY data/processed/faiss_e5_transcripts.index /app/data/processed/faiss_e5_transcripts.index

# Соберём колёса
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Финальный образ
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Добавим пользователя
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Копируем зависимости
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Установим зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && pip install --no-cache /wheels/* \
    && rm -rf /var/lib/apt/lists/*

# Скопируем код приложения
COPY src/ ./src/

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Запуск от непривилегированного пользователя
USER appuser

# Откроем порт
EXPOSE ${PORT}


# Запуск приложения
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
