# shai.studio - Generative Image Platform

Контейнеризованная платформа для генерации изображений с FastAPI, React и PostgreSQL.

## Архитектура

- **postgres** — база данных (порт 5432)
- **fastapi_api** — REST API (порт 8000)
- **frontend_app** — React SPA через Nginx (порт 3000)

## Быстрый старт

1. Создайте локальный `.env` в корне проекта:
```bash
cat > .env << 'EOF'
POSTGRES_USER=shai
POSTGRES_PASSWORD=change-me-strong-password
POSTGRES_DB=shai_db
DATABASE_URL=postgresql+psycopg2://shai:change-me-strong-password@postgres:5432/shai_db
JWT_SECRET=change-me-long-random-secret
JWT_EXPIRES_MIN=43200
EOF
```

2. Заполните переменные в `.env` (минимум `POSTGRES_PASSWORD`, `DATABASE_URL`, `JWT_SECRET`).

3. Запустите все сервисы:
```bash
docker-compose up --build
```

4. Откройте в браузере:
- **Фронтенд**: http://localhost или http://localhost:3000 (оба варианта работают)
- **API Swagger**: http://localhost:8000/docs

## Использование

1. На странице Auth введите любые креды (мок-авторизация)
2. Перейдите на вкладку Generation
3. Введите промпт и нажмите Generate
4. Через 5 секунд появится сгенерированное изображение (mock)

## База данных

Все задачи сохраняются в PostgreSQL. Для просмотра через DBeaver:

- **Host**: localhost
- **Port**: 5432
- **Database**: shai_db
- **User**: значение `POSTGRES_USER` из `.env`
- **Password**: значение `POSTGRES_PASSWORD` из `.env`

Таблица `tasks` создаётся автоматически при первом запуске.

## Настройка переменных окружения

Для запуска создайте `.env` и заполните необходимые переменные:

1. **Для Docker Compose**:
   ```
   POSTGRES_USER=shai
   POSTGRES_PASSWORD=<strong-password>
   POSTGRES_DB=shai_db
   DATABASE_URL=postgresql+psycopg2://shai:<strong-password>@postgres:5432/shai_db
   JWT_SECRET=<long-random-secret>
   ```

2. **Для локальной разработки** (если запускаете FastAPI не через Compose): можно также положить `.env` в `fastapi_api/`.
   ```
   # Минимум:
   JWT_SECRET=change-me-long-random-secret
   DATABASE_URL=postgresql+psycopg2://shai:<password>@localhost:5432/shai_db
   ```

## API Endpoints

- `POST /api/auth/register` — регистрация (email, name, password)
- `POST /api/auth/login` — авторизация (email, password) -> JWT access_token
- `POST /api/generate/image` — генерация изображения (требует `prompt` и `steps`, JWT)
- `POST /api/generate/video` — генерация видео (требует `prompt` и `steps`, JWT)
- `GET /api/tasks` — список последних задач


