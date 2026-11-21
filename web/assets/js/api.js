// The ONLY place the frontend talks to the server. Every number on screen comes through here.
// GET responses are cached for the session: the API is stateless, so (path + params) always gives the
// same answer. That keeps repeat views free (cost control) and makes the language toggle instant.
import { t } from "./i18n.js";

const cache = new Map();

export class ApiError extends Error {
  constructor(message, { offline = false, status = 0 } = {}) {
    super(message);
    this.offline = offline;
    this.status = status;
  }
}

async function request(path, init) {
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(t("state.offline"), { offline: true });
  }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) {
    const offline = res.status >= 502;   // proxy says the app is down
    throw new ApiError(offline ? t("state.offline") : (body && body.error) || t("state.error_title"), { offline, status: res.status });
  }
  return body;
}

export function get(path, params = {}) {
  const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ""));
  const url = qs.toString() ? `${path}?${qs}` : path;
  if (!cache.has(url)) {
    const p = request(url).catch((e) => { cache.delete(url); throw e; });   // never cache failures
    cache.set(url, p);
  }
  return cache.get(url);
}

export function post(path, body) {
  return request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

export const api = {
  meta: () => get("/api/meta"),
  forecast: (ward_id, as_of, horizon) => get("/api/forecast", { ward_id, as_of, horizon }),
  ranking: (as_of, horizon, city) => get("/api/ranking", { as_of, horizon, city }),
  warnings: (as_of, city) => get("/api/warnings", { as_of, city }),
  metrics: () => get("/api/metrics"),
  lagProfile: (city) => get("/api/lag_profile", { city }),
  history: (ward_id, horizon) => get("/api/history", { ward_id, horizon }),
  whatIf: (body) => post("/api/what_if", body),
};
