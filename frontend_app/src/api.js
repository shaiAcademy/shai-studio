import axios from "axios";

/**
 * Если ты запускаешь через docker-compose с nginx, то:
 * - фронт доступен на http://localhost:3000
 * - nginx проксирует /api -> fastapi_api:8000
 * Тогда API_BASE = "/api" идеально.
 *
 * Если хочешь напрямую указывать бек:
 * VITE_API_BASE="http://localhost:8000/api"
 */
const API_BASE = "http://localhost:8000/api";

export const login = async (username, password) => {
  const res = await axios.post(`${API_BASE}/auth/login`, {
    email: username,
    password,
  });
  return res.data;
};

export const register = async ({ email, name, password }) => {
  const res = await axios.post(`${API_BASE}/auth/register`, {
    email,
    name,
    password,
  });
  return res.data;
};

export const generateImage = async ({ prompt, steps = 30, token }) => {
  const res = await axios.post(
      `${API_BASE}/generate/image`,
      { prompt, steps },
      { headers: token ? { Authorization: `Bearer ${token}` } : {} }
  );
  return res.data; // ожидаем { id: "...", status: "IN_QUEUE" ... }
};

export const generateVideo = async ({ prompt, steps = 30, token }) => {
  const res = await axios.post(
      `${API_BASE}/generate/video`,
      { prompt, steps },
      { headers: token ? { Authorization: `Bearer ${token}` } : {} }
  );
  return res.data;
};

export const getRunpodStatus = async ({ taskId, token }) => {
  const res = await axios.get(`${API_BASE}/generate/status/${taskId}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  return res.data;
};

/**
 * Универсальная функция:
 * - запускает image/video
 * - поллит статус
 * - возвращает { taskId, kind, mediaUrl } когда готово
 */
export async function runAndWait({ kind, prompt, steps = 30, token, pollMs = 2000, timeoutMs = 10 * 60 * 1000 }) {
  const started =
      kind === "video"
          ? await generateVideo({ prompt, steps, token })
          : await generateImage({ prompt, steps, token });

  // RunPod обычно возвращает id в поле "id"
  const taskId = started.id || started.taskId || started.jobId;
  if (!taskId) {
    throw new Error(`No task id returned from backend. response=${JSON.stringify(started)}`);
  }

  const start = Date.now();

  while (Date.now() - start < timeoutMs) {
    const st = await getRunpodStatus({ taskId, token });

    // backend у тебя должен добавлять output.media_url когда COMPLETED
    // пример: st = { status: "COMPLETED", output: { media_key, media_url, ... } }
    const status = (st.status || "").toUpperCase();

    if (status === "COMPLETED") {
      const mediaUrl = st?.output?.media_url;
      if (!mediaUrl) {
        throw new Error(
            `Completed, but backend returned no media_url. output=${JSON.stringify(st?.output || st)}`
        );
      }
      return { taskId, kind, mediaUrl, raw: st };
    }

    if (status === "FAILED" || status === "CANCELLED") {
      throw new Error(`Task ${taskId} failed. status=${JSON.stringify(st)}`);
    }

    await new Promise((r) => setTimeout(r, pollMs));
  }

  throw new Error(`Timeout waiting for task ${taskId}`);
}
