import { useEffect, useMemo, useRef, useState } from "react";
import { Document, Packer, Paragraph, TextRun } from "docx";
import { createJob, deleteJob, fetchJob, fetchJobs, fetchStats, getJobAudioUrl } from "./api";

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

const POLL_INTERVAL_MS = 5000;

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

function formatTimestampHms(seconds) {
  const totalSec = Math.max(0, Math.round(Number(seconds) || 0));
  const sec = String(totalSec % 60).padStart(2, "0");
  const totalMin = Math.floor(totalSec / 60);
  const min = String(totalMin % 60).padStart(2, "0");
  const hrs = String(Math.floor(totalMin / 60)).padStart(2, "0");
  return `${hrs}:${min}:${sec}`;
}

function buildWordFilename(originalFilename) {
  const baseName = (originalFilename || "transcript").replace(/\.[^/.]+$/, "").trim();
  const safeName = baseName
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, "_")
    .replace(/\s+/g, " ")
    .trim();
  return `${safeName || "transcript"}_transcript.docx`;
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

function formatHourlySpeed(value) {
  if (typeof value !== "number" || Number.isNaN(value) || value < 0) {
    return "n/a";
  }
  return `${formatDurationMs(value)} / час аудио`;
}

function dateValueToIso(dateValue, endOfDay = false) {
  if (!dateValue) {
    return null;
  }
  const date = new Date(`${dateValue}T00:00:00`);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  if (endOfDay) {
    date.setHours(23, 59, 59, 999);
  }
  return date.toISOString();
}

function formatWhisperSettings(job) {
  const model = job?.whisper_model_size || "n/a";
  const cpuThreads =
    typeof job?.whisper_cpu_threads === "number" ? job.whisper_cpu_threads : "n/a";
  const beamSize =
    typeof job?.whisper_beam_size === "number" ? job.whisper_beam_size : "n/a";
  return `Model: ${model} | CPU_THREADS: ${cpuThreads} | BEAM_SIZE: ${beamSize}`;
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
  const [activeTab, setActiveTab] = useState("jobs");
  const [jobs, setJobs] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedJob, setSelectedJob] = useState(null);
  const [audioVariant, setAudioVariant] = useState("original");
  const [file, setFile] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [isJobsLoading, setIsJobsLoading] = useState(true);
  const [isJobLoading, setIsJobLoading] = useState(false);
  const [jobsError, setJobsError] = useState("");
  const [jobError, setJobError] = useState("");
  const [jobSuccess, setJobSuccess] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");
  const [statsFrom, setStatsFrom] = useState("");
  const [statsTo, setStatsTo] = useState("");
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState("");
  const audioRef = useRef(null);

  const selectedJobFromList = useMemo(
    () => jobs.find((job) => job.id === selectedId) || null,
    [jobs, selectedId]
  );
  const selectedJobStatusLabel =
    STATUS_LABELS[selectedJob?.status] || selectedJob?.status || "Неизвестно";
  const selectedJobStageLabel = STAGE_LABELS[selectedJob?.stage] || selectedJob?.stage;
  const selectedJobSupportsPreparedAudio = Boolean(selectedJob?.prepared_audio_url);
  const selectedAudioUrl =
    selectedJob &&
    getJobAudioUrl(
      selectedJob.id,
      audioVariant === "prepared" && selectedJobSupportsPreparedAudio ? "prepared" : "original"
    );

  async function refreshJobs({ silent = false } = {}) {
    if (!silent) {
      setIsJobsLoading(true);
    }
    try {
      const data = await fetchJobs();
      setJobsError("");
      setJobs(data);
      if (!selectedId && data.length > 0) {
        setSelectedId(data[0].id);
      } else if (selectedId && !data.some((job) => job.id === selectedId)) {
        setSelectedId(data[0]?.id || null);
      }
    } catch (err) {
      setJobsError(err.message);
    } finally {
      if (!silent) {
        setIsJobsLoading(false);
      }
    }
  }

  useEffect(() => {
    let mounted = true;
    refreshJobs();
    const intervalId = setInterval(() => {
      if (mounted) {
        refreshJobs({ silent: true });
      }
    }, POLL_INTERVAL_MS);

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

    setAudioVariant("original");
    setJobSuccess("");
    let mounted = true;
    async function loadJob(silent = false) {
      if (mounted && !silent) {
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
        if (mounted && !silent) {
          setIsJobLoading(false);
        }
      }
    }

    loadJob();
    const intervalId = setInterval(() => loadJob(true), POLL_INTERVAL_MS);
    return () => {
      mounted = false;
      clearInterval(intervalId);
    };
  }, [selectedId]);

  async function loadStats() {
    setStatsLoading(true);
    try {
      const data = await fetchStats({
        from: dateValueToIso(statsFrom),
        to: dateValueToIso(statsTo, true),
      });
      setStats(data);
      setStatsError("");
    } catch (err) {
      setStatsError(err.message);
    } finally {
      setStatsLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "stats") {
      loadStats();
    }
  }, [activeTab]);

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) {
      setUploadError("Сначала выберите аудиофайл.");
      setUploadSuccess("");
      return;
    }
    setUploadError("");
    setUploadSuccess("");
    setJobSuccess("");
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

  async function onDeleteSelectedJob() {
    const currentId = selectedJob?.id || selectedId;
    if (!currentId || isDeleting) {
      return;
    }
    const accepted = window.confirm("Удалить выбранную заявку?");
    if (!accepted) {
      return;
    }
    setIsDeleting(true);
    setJobError("");
    setJobSuccess("");
    try {
      const result = await deleteJob(currentId);
      if (result.status === "pending_delete") {
        setJobSuccess(result.message);
        const [list, detail] = await Promise.all([fetchJobs(), fetchJob(currentId)]);
        setJobs(list);
        setSelectedJob(detail);
      } else {
        setJobSuccess(result.message);
        const list = await fetchJobs();
        setJobs(list);
        if (selectedId === currentId) {
          setSelectedId(list[0]?.id || null);
        }
      }
    } catch (err) {
      setJobError(err.message);
    } finally {
      setIsDeleting(false);
    }
  }

  async function onDownloadWord() {
    if (!selectedJob || selectedJob.status !== "done") {
      return;
    }
    const segments = selectedJob.segments || [];
    if (segments.length === 0) {
      setJobError("Для этой задачи нет сегментов для экспорта.");
      return;
    }

    try {
      setJobError("");
      const docParagraphs = [];
      segments.forEach((segment, index) => {
        docParagraphs.push(
          new Paragraph({
            children: [
              new TextRun(
                `(${formatTimestampHms(segment.start_sec)}-${formatTimestampHms(segment.end_sec)})`
              ),
            ],
          }),
          new Paragraph({
            children: [new TextRun(segment.text || "")],
          })
        );

        if (index < segments.length - 1) {
          docParagraphs.push(new Paragraph(""));
        }
      });

      const doc = new Document({
        sections: [
          {
            children: docParagraphs,
          },
        ],
      });

      const blob = await Packer.toBlob(doc);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = buildWordFilename(selectedJob.original_filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      setJobError(`Не удалось сформировать Word-файл: ${error?.message || "неизвестная ошибка"}`);
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
        <div className="topbar-chip">API polling: 5s</div>
      </header>

      <section className="card tabs-card">
        <div className="tabs">
          <button
            type="button"
            className={activeTab === "jobs" ? "tab-button active" : "tab-button"}
            onClick={() => setActiveTab("jobs")}
          >
            Очередь
          </button>
          <button
            type="button"
            className={activeTab === "stats" ? "tab-button active" : "tab-button"}
            onClick={() => setActiveTab("stats")}
          >
            Статистика
          </button>
        </div>
      </section>

      <section className="card upload-card">
        <h2>Новая транскрибация</h2>
        <p className="section-caption">Поддерживаются аудиофайлы и видео-контейнеры с аудиодорожкой. После загрузки задача появится в списке слева.</p>
        <form onSubmit={onSubmit} className="upload-form">
          <label className="file-control">
            <span>Аудиофайл</span>
            <input
              type="file"
              accept="audio/*,video/*,.mkv,.avi,.mov"
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

      {activeTab === "jobs" && (
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
                  <span className="meta">{formatWhisperSettings(job)}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="card result-card">
          <h2>Результат</h2>
          {jobError && <p className="inline-message is-danger">{jobError}</p>}
          {jobSuccess && <p className="inline-message is-success">{jobSuccess}</p>}
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
              <p>
                <strong>Настройки:</strong> {formatWhisperSettings(selectedJob)}
              </p>
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
              <div className="result-actions">
                <button
                  type="button"
                  className="secondary-button danger-button"
                  onClick={onDeleteSelectedJob}
                  disabled={isDeleting}
                >
                  {isDeleting ? "Удаление..." : "Удалить заявку"}
                </button>
              </div>
              {selectedJob.delete_requested && (
                <p className="loading-state">
                  Удаление запрошено. Заявка будет удалена после завершения текущего этапа.
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
                  <p>
                    <strong>Предобработка:</strong> {formatDurationMs(selectedJob.preprocess_duration_ms)}
                  </p>
                  <p>
                    <strong>Общее время:</strong> {formatDurationMs(selectedJob.processing_duration_ms)}
                  </p>
                  <div className="result-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={onDownloadWord}
                      disabled={selectedJob.segments.length === 0}
                    >
                      Скачать Word (.docx)
                    </button>
                  </div>
                  <label className="audio-variant-control">
                    <span>Источник аудио</span>
                    <select
                      value={
                        audioVariant === "prepared" && !selectedJobSupportsPreparedAudio
                          ? "original"
                          : audioVariant
                      }
                      onChange={(event) => setAudioVariant(event.target.value)}
                    >
                      <option value="original">Оригинал</option>
                      <option value="prepared" disabled={!selectedJobSupportsPreparedAudio}>
                        Обработанное
                      </option>
                    </select>
                  </label>
                  <audio
                    key={`${selectedJob.id}-${audioVariant}`}
                    ref={audioRef}
                    controls
                    preload="metadata"
                    className="audio-player"
                    src={selectedAudioUrl}
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
      )}

      {activeTab === "stats" && (
        <section className="card stats-card">
          <div className="section-head">
            <h2>Статистика обработки</h2>
            <button type="button" className="secondary-button" onClick={loadStats} disabled={statsLoading}>
              {statsLoading ? "Обновление..." : "Обновить"}
            </button>
          </div>
          <div className="stats-filters">
            <label>
              <span>Дата от</span>
              <input
                type="date"
                value={statsFrom}
                onChange={(event) => setStatsFrom(event.target.value)}
              />
            </label>
            <label>
              <span>Дата до</span>
              <input type="date" value={statsTo} onChange={(event) => setStatsTo(event.target.value)} />
            </label>
          </div>
          {statsError && <p className="inline-message is-danger">{statsError}</p>}
          {!statsLoading && stats && (
            <>
              <p className="meta">Завершенных заявок в диапазоне: {stats.completed_jobs}</p>
              <div className="stats-grid">
                <article className="stats-tile">
                  <h3>Средняя скорость</h3>
                  <p><strong>Итог:</strong> {formatHourlySpeed(stats.average.total_ms)}</p>
                  <p><strong>Предобработка:</strong> {formatHourlySpeed(stats.average.preprocess_ms)}</p>
                  <p><strong>Транскрибация:</strong> {formatHourlySpeed(stats.average.transcribe_ms)}</p>
                </article>
                <article className="stats-tile">
                  <h3>Самое быстрое</h3>
                  <p className="meta">{stats.fastest.original_filename || "n/a"}</p>
                  <p><strong>Итог:</strong> {formatHourlySpeed(stats.fastest.total_ms)}</p>
                  <p><strong>Предобработка:</strong> {formatHourlySpeed(stats.fastest.preprocess_ms)}</p>
                  <p><strong>Транскрибация:</strong> {formatHourlySpeed(stats.fastest.transcribe_ms)}</p>
                </article>
                <article className="stats-tile">
                  <h3>Самое долгое</h3>
                  <p className="meta">{stats.slowest.original_filename || "n/a"}</p>
                  <p><strong>Итог:</strong> {formatHourlySpeed(stats.slowest.total_ms)}</p>
                  <p><strong>Предобработка:</strong> {formatHourlySpeed(stats.slowest.preprocess_ms)}</p>
                  <p><strong>Транскрибация:</strong> {formatHourlySpeed(stats.slowest.transcribe_ms)}</p>
                </article>
              </div>
            </>
          )}
          {statsLoading && <p className="loading-state">Загружаем статистику...</p>}
        </section>
      )}
    </div>
  );
}

export default App;
