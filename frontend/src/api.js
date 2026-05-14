const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const DEFAULT_TIMEOUT_MS = 15000;
const RETRY_DELAYS_MS = [300, 900];

export class ApiError extends Error {
  constructor(message, { code = "REQUEST_FAILED", status = null, details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

function toDetailMessage(detail) {
  if (Array.isArray(detail)) {
    const first = detail[0];
    if (!first) {
      return "";
    }
    if (typeof first === "string") {
      return first;
    }
    if (typeof first === "object" && first.msg) {
      return String(first.msg);
    }
    return JSON.stringify(first);
  }
  if (typeof detail === "object" && detail !== null) {
    if (detail.message) {
      return String(detail.message);
    }
    return JSON.stringify(detail);
  }
  return typeof detail === "string" ? detail : "";
}

function isTransientError(error) {
  if (error instanceof ApiError && error.status !== null) {
    return error.status >= 500 || error.status === 429;
  }
  return (
    error instanceof ApiError &&
    (error.code === "NETWORK_ERROR" || error.code === "TIMEOUT")
  );
}

function buildApiError(status, body) {
  if (body && typeof body === "object") {
    const message =
      typeof body.message === "string" && body.message
        ? body.message
        : toDetailMessage(body.detail) || `Ошибка запроса (код ${status})`;
    return new ApiError(message, {
      code: body.code || "HTTP_ERROR",
      status,
      details: body.details ?? body.detail ?? null,
    });
  }
  return new ApiError(`Ошибка запроса (код ${status})`, {
    code: "HTTP_ERROR",
    status,
  });
}

async function parseJson(response) {
  if (!response.ok) {
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    throw buildApiError(response.status, body);
  }
  return response.json();
}

async function request(path, init = {}, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });
    return await parseJson(response);
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new ApiError("Превышено время ожидания ответа сервера.", {
        code: "TIMEOUT",
      });
    }
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError("Не удалось выполнить запрос. Проверьте подключение к сети.", {
      code: "NETWORK_ERROR",
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

async function requestWithRetry(path, init = {}, options = {}) {
  const retryCount = options.retryCount ?? 0;
  const retryDelays = options.retryDelays ?? RETRY_DELAYS_MS;
  let attempt = 0;

  while (true) {
    try {
      return await request(path, init, options);
    } catch (error) {
      const canRetry = attempt < retryCount && isTransientError(error);
      if (!canRetry) {
        throw error;
      }
      await new Promise((resolve) => {
        setTimeout(resolve, retryDelays[attempt] ?? retryDelays[retryDelays.length - 1] ?? 500);
      });
      attempt += 1;
    }
  }
}

export async function createJob(file) {
  const formData = new FormData();
  formData.append("file", file);
  return requestWithRetry(
    "/api/jobs",
    {
      method: "POST",
      body: formData,
    },
    {
      timeoutMs: 300000,
      retryCount: 0,
    }
  );
}

export async function fetchJobs() {
  return requestWithRetry("/api/jobs", undefined, { retryCount: 2 });
}

export async function fetchJob(jobId) {
  return requestWithRetry(`/api/jobs/${jobId}`, undefined, { retryCount: 2 });
}

export async function deleteJob(jobId) {
  return requestWithRetry(
    `/api/jobs/${jobId}`,
    {
      method: "DELETE",
    },
    { retryCount: 1 }
  );
}

export function getJobAudioUrl(jobId, variant = "original") {
  const params = new URLSearchParams({ variant });
  return `${API_BASE_URL}/api/jobs/${jobId}/audio?${params.toString()}`;
}
