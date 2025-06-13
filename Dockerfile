FROM python:3.11-slim AS builder

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
    && rm -rf /var/lib/apt/lists/*

# Скопируем зависимости
COPY requirements.txt .

# Соберём колёса
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Финальный образ
FROM python:3.11-slim

# Добавим пользователя
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Копируем зависимости
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Установим зависимости
RUN pip install --no-cache /wheels/*

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
