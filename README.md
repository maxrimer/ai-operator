# ai-operator

Repository for AI-operator project for 2025 Bereke AI-hackathon

## Инструкция по развертыванию с помощью Docker Compose

### Предварительные требования

- Установленный Docker
- Установленный Docker Compose
- NVIDIA GPU с установленными драйверами (для GPU-ускорения)

### Шаги по развертыванию

1. Клонируйте репозиторий:

```bash
git clone <url-репозитория>
cd ai-operator
```

2. Соберите и запустите контейнеры:

```bash
docker-compose up -d --build
```

Для запуска в фоновом режиме используйте:

```bash
docker-compose up -d --build
```

3. Проверьте работу приложения:

- Приложение будет доступно по адресу: http://localhost:8000
- API документация доступна по адресу: http://localhost:8000/docs

### Управление контейнерами

- Остановить контейнеры:

```bash
docker-compose down
```

- Просмотр логов:

```bash
docker-compose logs -f
```

- Перезапуск контейнеров:

```bash
docker-compose restart
```

### Особенности конфигурации

- Приложение запускается на порту 8000
- Используется Python 3.11
- Поддерживается GPU-ускорение через NVIDIA Container Toolkit
- Контейнер автоматически перезапускается при сбоях (restart: unless-stopped)
- Исходный код монтируется как volume для удобства разработки
