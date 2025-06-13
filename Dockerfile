FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

WORKDIR /app

# Установим переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Установим Python и зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Скопируем зависимости
COPY requirements.txt .
COPY data/processed/faiss_index.index /app/data/processed/faiss_index.index
COPY data/processed/faiss_e5_transcripts.index /app/data/processed/faiss_e5_transcripts.index

# Соберём колёса
RUN pip3 wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Финальный образ
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Добавим пользователя
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Установим Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Копируем зависимости
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Установим зависимости
RUN pip3 install --no-cache /wheels/*

# Скопируем код приложения
COPY src/ ./src/

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Запуск от непривилегированного пользователя
USER appuser

# Откроем порт
EXPOSE ${PORT}

# Запуск приложения
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
