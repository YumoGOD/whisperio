import { useEffect, useMemo, useRef, useState } from "react";
import { createJob, fetchJob, fetchJobs, getJobAudioUrl } from "./api";

const STATUS_LABELS = {
  queued: "В очереди",
  processing: "Обрабатывается",
  done: "Готово",
  failed: "Ошибка",
};

const STAGE_LABELS = {
  queued: "Ожидает запуска",
  preparing: "Подготовка",
  preprocessing: "Предобработка аудио",
  transcribing: "Распознавание речи",
  saving_segments: "Сохранение сегментов",
  completed: "Завершено",
  failed: "Сбой обработки",
};

function formatTimestamp(seconds) {
  const totalMs = Math.max(0, Math.round((seconds || 0) * 1000));
  const ms = String(totalMs % 1000).padStart(3, "0");
  const totalSec = Math.floor(totalMs / 1000);
  const sec = String(totalSec % 60).padStart(2, "0");
  const totalMin = Math.floor(totalSec / 60);
  const min = String(totalMin % 60).padStart(2, "0");
  const hrs = String(Math.floor(totalMin / 60)).padStart(2, "0");
  return `${hrs}:${min}:${sec}.${ms}`;
}

function formatDateTime(value) {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "n/a";
  }
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatDurationMs(value) {
  if (typeof value !== "number" || Number.isNaN(value) || value < 0) {
    return "n/a";
  }
  const totalSec = Math.round(value / 1000);
  const sec = String(totalSec % 60).padStart(2, "0");
  const min = String(Math.floor((totalSec / 60) % 60)).padStart(2, "0");
  const hrs = Math.floor(totalSec / 3600);
  return hrs > 0 ? `${hrs}:${min}:${sec}` : `${min}:${sec}`;
}

function getStatusClass(status) {
  if (status === "done") {
    return "is-success";
  }
  if (status === "failed") {
    return "is-danger";
  }
  if (status === "processing") {
    return "is-warning";
  }
  return "is-neutral";
}

function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedJob, setSelectedJob] = useState(null);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [isJobsLoading, setIsJobsLoading] = useState(true);
  const [isJobLoading, setIsJobLoading] = useState(false);
  const [jobsError, setJobsError] = useState("");
  const [jobError, setJobError] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");
  const audioRef = useRef(null);

  const selectedJobFromList = useMemo(
    () => jobs.find((job) => job.id === selectedId) || null,
    [jobs, selectedId]
  );
  const selectedJobStatusLabel =
    STATUS_LABELS[selectedJob?.status] || selectedJob?.status || "Неизвестно";
  const selectedJobStageLabel = STAGE_LABELS[selectedJob?.stage] || selectedJob?.stage;

  useEffect(() => {
    let mounted = true;

    async function loadJobs(silent = false) {
      if (!silent && mounted) {
        setIsJobsLoading(true);
      }
      try {
        const data = await fetchJobs();
        if (mounted) {
          setJobsError("");
          setJobs(data);
          if (!selectedId && data.length > 0) {
            setSelectedId(data[0].id);
          }
          if (selectedId && !data.some((job) => job.id === selectedId)) {
            setSelectedId(data[0]?.id || null);
          }
        }
      } catch (err) {
        if (mounted) {
          setJobsError(err.message);
        }
      } finally {
        if (mounted && !silent) {
          setIsJobsLoading(false);
        }
      }
    }

    loadJobs();
    const intervalId = setInterval(() => loadJobs(true), 2500);

    return () => {
      mounted = false;
      clearInterval(intervalId);
    };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedJob(null);
      return;
    }

    let mounted = true;
    async function loadJob() {
      if (mounted) {
        setIsJobLoading(true);
      }
      try {
        const data = await fetchJob(selectedId);
        if (mounted) {
          setSelectedJob(data);
          setJobError("");
        }
      } catch (err) {
        if (mounted) {
          setJobError(err.message);
        }
      } finally {
        if (mounted) {
          setIsJobLoading(false);
        }
      }
    }

    loadJob();
    const intervalId = setInterval(loadJob, 2500);
    return () => {
      mounted = false;
      clearInterval(intervalId);
    };
  }, [selectedId]);

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) {
      setUploadError("Сначала выберите аудиофайл.");
      setUploadSuccess("");
      return;
    }
    setUploadError("");
    setUploadSuccess("");
    setUploading(true);
    try {
      const { job_id: jobId } = await createJob(file);
      const list = await fetchJobs();
      setJobs(list);
      setSelectedId(jobId);
      setFile(null);
      setUploadSuccess("Файл успешно загружен. Транскрибация уже в очереди.");
      event.target.reset();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function onSegmentClick(segment) {
    const player = audioRef.current;
    if (!player) {
      return;
    }
    try {
      player.currentTime = Number(segment.start_sec || 0);
      await player.play();
    } catch {
      // Browser may block autoplay without direct interaction.
    }
  }

  return (
    <div className="page app-shell">
      <header className="topbar card">
        <div>
          <h1>WhisperIO</h1>
          <p className="subtitle">
            Современный desktop-first интерфейс для очереди транскрибаций и удобного чтения результата.
          </p>
        </div>
        <div className="topbar-chip">API polling: 2.5s</div>
      </header>

      <section className="card upload-card">
        <h2>Новая транскрибация</h2>
        <p className="section-caption">Поддерживаются популярные аудиоформаты. После загрузки задача появится в списке слева.</p>
        <form onSubmit={onSubmit} className="upload-form">
          <label className="file-control">
            <span>Аудиофайл</span>
            <input
              type="file"
              accept="audio/*"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </label>
          <button type="submit" disabled={uploading} className="primary-button">
            {uploading ? "Загрузка..." : "Загрузить"}
          </button>
        </form>
        {uploadError && <p className="inline-message is-danger">{uploadError}</p>}
        {uploadSuccess && <p className="inline-message is-success">{uploadSuccess}</p>}
      </section>

      <main className="grid content-grid">
        <section className="card jobs-card">
          <div className="section-head">
            <h2>Задачи</h2>
            <span className="counter">{jobs.length}</span>
          </div>
          {jobsError && <p className="inline-message is-danger">{jobsError}</p>}
          {isJobsLoading && (
            <div className="skeleton-list">
              <div className="skeleton-item" />
              <div className="skeleton-item" />
              <div className="skeleton-item" />
            </div>
          )}
          {!isJobsLoading && jobs.length === 0 && (
            <p className="empty-state">Пока нет задач. Загрузите первый аудиофайл.</p>
          )}
          <ul className="job-list">
            {jobs.map((job) => (
              <li key={job.id}>
                <button
                  className={job.id === selectedId ? "job-item active" : "job-item"}
                  type="button"
                  onClick={() => setSelectedId(job.id)}
                >
                  <div className="job-row">
                    <strong>{job.original_filename}</strong>
                    <span className={`status-pill ${getStatusClass(job.status)}`}>
                      {STATUS_LABELS[job.status] || job.status}
                    </span>
                  </div>
                  <span className="meta">
                    Этап: {STAGE_LABELS[job.stage] || job.stage || "Ожидание"}
                  </span>
                  {typeof job.progress === "number" && (
                    <div className="progress-wrap">
                      <span>Прогресс: {Math.round(job.progress)}%</span>
                      <div className="progress-track">
                        <div
                          className="progress-value"
                          style={{ width: `${Math.max(0, Math.min(100, Math.round(job.progress)))}%` }}
                        />
                      </div>
                    </div>
                  )}
                  <span className="meta">Создано: {formatDateTime(job.created_at)}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="card result-card">
          <h2>Результат</h2>
          {jobError && <p className="inline-message is-danger">{jobError}</p>}
          {!selectedId && <p className="empty-state">Выберите задачу, чтобы посмотреть детали.</p>}
          {selectedId && isJobLoading && <p className="loading-state">Загружаем детали задачи...</p>}
          {selectedJob && (
            <>
              <div className="result-head">
                <p>
                  <strong>Файл:</strong> {selectedJob.original_filename}
                </p>
                <span className={`status-pill ${getStatusClass(selectedJob.status)}`}>
                  {selectedJobStatusLabel}
                </span>
              </div>
              {selectedJob.stage && (
                <p>
                  <strong>Этап:</strong> {selectedJobStageLabel}
                </p>
              )}
              {typeof selectedJob.progress === "number" && (
                <p>
                  <strong>Прогресс:</strong> {Math.round(selectedJob.progress)}%
                </p>
              )}
              {selectedJob.status_message && (
                <p>
                  <strong>Детали:</strong> {selectedJob.status_message}
                </p>
              )}
              {selectedJob.error && (
                <p className="inline-message is-danger">
                  <strong>Ошибка:</strong> {selectedJob.error}
                </p>
              )}
              {selectedJob.error_code && (
                <p className="inline-message is-danger">
                  <strong>Код ошибки:</strong> {selectedJob.error_code}
                </p>
              )}
              {selectedJob.status !== "done" && (
                <p className="loading-state">Транскрибация выполняется в очереди.</p>
              )}
              {selectedJob.status === "done" && (
                <>
                  <p>
                    <strong>Длительность:</strong>{" "}
                    {formatTimestamp(selectedJob.duration_sec || 0)}
                  </p>
                  <p>
                    <strong>Время транскрибации:</strong>{" "}
                    {formatDurationMs(
                      selectedJob.transcribe_duration_ms ?? selectedJob.processing_duration_ms
                    )}
                  </p>
                  <audio
                    key={selectedJob.id}
                    ref={audioRef}
                    controls
                    preload="metadata"
                    className="audio-player"
                    src={getJobAudioUrl(selectedJob.id)}
                  />
                  <div className="segments">
                    {selectedJob.segments.length === 0 && <p>Текстовые сегменты отсутствуют.</p>}
                    {selectedJob.segments.map((segment) => (
                      <div key={`${segment.idx}-${segment.start_sec}`} className="segment">
                        <button
                          type="button"
                          className="segment-button"
                          onClick={() => onSegmentClick(segment)}
                        >
                          <span className="stamp">
                            {formatTimestamp(segment.start_sec)} -{" "}
                            {formatTimestamp(segment.end_sec)}
                          </span>
                          <p>{segment.text}</p>
                        </button>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
          {selectedId && !selectedJob && selectedJobFromList?.status && !isJobLoading && (
            <p className="loading-state">
              Текущий статус: {STATUS_LABELS[selectedJobFromList.status] || selectedJobFromList.status}
            </p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
