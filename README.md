# Local Transcriber

Local app for transcribing long and noisy lecture recordings with timings. **A NVIDIA GPU is required** — the Docker image ships CUDA/cuDNN and the worker runs faster-whisper on CUDA only. Stack: FastAPI, SQLite, local files, ffmpeg, faster-whisper. No Redis, Celery, PostgreSQL, RabbitMQ or cloud service is required.

## Features

- Upload audio/video files readable by ffmpeg, including `mp3`, `wav`, `ogg`, `m4a`, `flac`, `webm`, `mp4`.
- Store media and artifacts under `./data/uploads`, `./data/work`, `./data/transcripts`, `./data/logs`.
- Run API and worker as separate Docker Compose services.
- Resume unfinished work after worker restart by returning stale `running` jobs to `pending`.
- Preprocess to mono 16 kHz WAV, optionally with ffmpeg loudness normalization.
- Chunk long recordings with overlap to reduce word loss at boundaries.
- Export `txt`, `json`, `srt`, `vtt` and timestamped `docx`.
- Review the source audio in the browser and click timestamped transcript segments to seek playback.
- Use a corporate glossary and per-job context to improve brand, product, person and abbreviation recognition.
- Compare `accuracy_first` and `speed_balanced` profiles.
- Benchmark one file and print real-time factor.

## GPU and Docker

The [Dockerfile](Dockerfile) uses `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`; inference is fixed to **CUDA** in code (no CPU path). On the host you need a recent **NVIDIA driver** and the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**. In [docker-compose.yml](docker-compose.yml), the `worker` service has `gpus: all`; the API service does not use a GPU.

On **Windows**, use Docker Desktop with **WSL2** and GPU support in the distro; native Windows containers with GPU are limited.

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:8000`, upload a file, and watch the job list. The first run downloads the faster-whisper model into `./data/models`, so it can be much slower than later runs.

To confirm the worker sees a GPU:

```bash
docker compose run --rm worker nvidia-smi
```

If the server has restricted internet or Hugging Face SSL errors, download the model once and use it as a local path:

```bash
docker compose run --rm worker python scripts/download_model.py large-v3 --output-dir /app/data/models/faster-whisper-large-v3
```

Then set this in `.env`:

```env
WHISPER_MODEL=/app/data/models/faster-whisper-large-v3
```

## API

```bash
curl -F "file=@lecture.mp3" -F "profile=accuracy_first" http://localhost:8000/api/jobs
curl http://localhost:8000/api/jobs
curl http://localhost:8000/api/jobs/<job_id>
curl -OJ http://localhost:8000/api/jobs/<job_id>/audio
curl -OJ http://localhost:8000/api/jobs/<job_id>/download/srt
curl -OJ http://localhost:8000/api/jobs/<job_id>/download/docx
```

Download formats:

- `GET /api/jobs/{job_id}/download/txt`
- `GET /api/jobs/{job_id}/download/json`
- `GET /api/jobs/{job_id}/download/srt`
- `GET /api/jobs/{job_id}/download/vtt`
- `GET /api/jobs/{job_id}/download/docx`

## Configuration

All important settings are controlled through `.env`.

Important defaults:

- `WHISPER_MODEL=large-v3`
- `WHISPER_COMPUTE_TYPE=float16` (optional: `int8_float16` to reduce VRAM)
- `WHISPER_LANGUAGE=ru`
- `WHISPER_TASK=transcribe`
- `WORKER_CONCURRENCY=1` (on one GPU usually leave at `1`)
- `DEFAULT_PROFILE=accuracy_first`
- `CHUNK_SECONDS=1800`
- `CHUNK_OVERLAP_SECONDS=15`
- `MAX_UPLOAD_MB=5120`
- `GLOSSARY_PATH=/app/data/glossary/global.yml`
- `GLOSSARY_ENABLE_HOTWORDS=false`
- `GLOSSARY_ENABLE_HARD_NORMALIZATION=true`

Device is always **CUDA** in the application; choosing a GPU is done with `CUDA_VISIBLE_DEVICES` on the host or in Compose if needed.

`WHISPER_MODEL` can be either a model alias like `large-v3` or a local CTranslate2 model directory, for example `/app/data/models/faster-whisper-large-v3`. A local path prevents runtime downloads and is recommended for offline or unstable networks.

`WHISPER_LANGUAGE=ru` fixes language detection to Russian, which is recommended for long Russian lectures because chunks will not randomly switch language. Set it to another ISO language code if needed, or leave it empty only if automatic language detection is required.

## Glossary And Job Context

The global corporate glossary is stored in `./data/glossary/global.yml` and is applied to every job. It supports two modes:

- `soft`: term can be used in the Whisper prompt when relevant to the job context, but the recognized text is not rewritten.
- `hard`: term can be used in the prompt and is also normalized after recognition using regex replacements.

`hotwords` are disabled by default because they can be too aggressive on long noisy Russian recordings and may cause repeated glossary hallucinations at the beginning of a chunk. Enable them only after testing a short sample:

```env
GLOSSARY_ENABLE_HOTWORDS=false
GLOSSARY_PROMPT_MAX_CHARS=700
GLOSSARY_REPETITION_COMPRESSION_THRESHOLD=4.0
```

Example:

```yaml
terms:
  - canonical: "Bauer"
    category: "brand"
    mode: "hard"
    spoken_forms:
      - "Бауэр"
      - "Бауер"
    description: "Bauer — название компании, произносится как Бауэр."
    replacements:
      - from: "\\b[Бб]ауэр\\b"
        to: "Bauer"
      - from: "\\b[Бб]ауер\\b"
        to: "Bauer"
```

When creating a job, you can also provide:

- `audio_type`: type of recording, for example `лекция`, `планерка`, `презентация`.
- `audio_context`: short description of what is in the audio.
- `expected_content`: expected topics, blocks and structure.
- `dynamic_terms`: extra terms for this job only.

Dynamic terms format:

```text
Bauer Juicer Pro | hard | Бауэр джусер про, джусер про, соковыжималка Бауэр
Bauer Lounge Massage | hard | Бауэр лаундж массаж, лаундж мессаж, массажная накидка
Дмитрий Канищев | soft | Канищев, Дмитрий Канищев
```

Glossary diagnostics are saved in `diagnostics.json` and shown on the job page: prompt, term counts, dynamic terms, hard replacement counts, repeated glossary hallucination drops and the terms used for the job.

## Profiles

`accuracy_first` is the default for noisy lectures:

- uses `large-v3` with beam search;
- disables VAD by default to minimize speech loss;
- uses overlap chunking;
- sets `condition_on_previous_text=false` to reduce long-recording cascade errors;
- favors recall over speed.

`speed_balanced` is for cleaner files:

- lower beam settings;
- cautious VAD enabled;
- still uses overlap chunking.

VAD can be overridden globally:

```env
VAD_FILTER=false
VAD_THRESHOLD=0.35
VAD_MIN_SILENCE_MS=1200
VAD_SPEECH_PAD_MS=600
```

## Benchmark

Run inside Compose:

```bash
docker compose run --rm worker python scripts/benchmark.py /app/data/uploads/example.mp3 --profile accuracy_first
```

Or copy a local file into `./data/uploads` first and point the script to `/app/data/uploads/<file>`. The script prints audio duration, elapsed time and real-time factor.

## Artifacts

For each job:

- uploaded source: `./data/uploads/<job_id>_<filename>`
- prepared WAV: `./data/work/<job_id>/prepared_16k_mono.wav`
- chunks: `./data/work/<job_id>/chunks`
- exports: `./data/transcripts/<job_id>/<job_id>.txt|json|srt|vtt|docx`
- diagnostics: `./data/transcripts/<job_id>/diagnostics.json`
- logs: `./data/logs/app.log`

These files are intentionally kept so you can inspect where speech may have been lost.

## Notes

- SQLite is used with WAL mode and short write transactions. This is enough for a local API plus worker setup and can be scaled later by adding more worker services carefully.
- The app accepts any extension by default with `ALLOWED_EXTENSIONS=*`; ffmpeg is the real decoder gate. Set a comma-separated allowlist if you want stricter uploads.
- Very large files are limited by `MAX_UPLOAD_MB`.
