# Whisper Transcription Service

Веб-сервис: загрузка аудио/видео, очередь задач в SQLite, воркер с faster-whisper, UI на React.

## Требования

- **Python 3.11**, **ffmpeg** в `PATH`
- **Node.js** 20+ (для UI)

## Без Docker

1. Склонируйте репозиторий, в корне:

   ```bash
   cp .env.example .env
   ```

2. Python (виртуальное окружение из корня репозитория):

   ```bash
   python -m venv .venv
   ```

   Активация: Windows — `.venv\Scripts\activate`, Linux/macOS — `source .venv/bin/activate`.

   PyTorch ставьте **до** `requirements.txt`, иначе подтянется несовместимый `torchaudio` (см. комментарии в `backend/requirements.txt`). Пример для CPU:

   - **Без GPU (рекомендуется для dev):** в `.env` задайте `WHISPER_DEVICE=cpu` и `WHISPER_COMPUTE_TYPE=int8` (или `float32`). Затем:

   ```bash
   pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
   pip install -r backend/requirements.txt
   ```

   - **С GPU (CUDA 12.x):** в `.env` обычно `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`. Затем:

   ```bash
   pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -r backend/requirements.txt
   ```

3. Запуск — **три терминала**, рабочая директория для API и воркера: `backend`:

   ```bash
   cd backend
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   ```bash
   cd backend
   python -m app.worker
   ```

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Конфиг читает `.env` из **корня репозитория**. UI: [http://localhost:5173](http://localhost:5173), API проксируется на порт 8000.

## Docker (API + worker)

В корне репозитория создайте `.env` (скопируйте из `.env.example`). В Compose для сервисов **принудительно** заданы `WHISPER_DEVICE=cpu` и `WHISPER_COMPUTE_TYPE=int8`, чтобы совпасть с CPU-образом (строки `cuda` из `.env` на воркер в контейнере не подходят).

```bash
docker compose up --build
```

API: [http://localhost:8000](http://localhost:8000). Фронт при этом запускайте локально (`cd frontend && npm run dev`) или отдавайте статику отдельно.

Чтобы использовать **GPU в контейнере**, нужны NVIDIA Container Toolkit и отдельный Dockerfile с CUDA и torch (cu121); текущий `Dockerfile` рассчитан только на CPU.

## Полезное

- **systemd:** можно оформить три юнита `ExecStart` аналогично командам выше (каталог `WorkingDirectory` для API/воркера — `.../backend`, переменные окружения — из `.env` в корне проекта).
- Без видеокарты задайте в `.env`: `WHISPER_DEVICE=cpu` — всё должно работать медленнее, но без CUDA.
