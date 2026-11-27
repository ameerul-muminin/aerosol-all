// The ONLY place the frontend talks to the server. Every number on screen comes through here.
// GET responses are cached for the session: the API is stateless, so (path + params) always gives the
// same answer. That keeps repeat views free (cost control) and makes the language toggle instant.
import { t } from "./i18n.js";

const cache = new Map();
/** How many GET answers this browser reused from memory instead of asking the server (Cost & speed tab). */
export const browserCache = { reused: 0, asked: 0 };

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
  const qs = clean(params);
  const url = qs.toString() ? `${path}?${qs}` : path;
  if (cache.has(url)) browserCache.reused += 1;
  else {
    browserCache.asked += 1;
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
  efficiency: () => request("/api/efficiency"),   // live counters: never cached
  metrics: () => get("/api/metrics"),
  lagProfile: (city) => get("/api/lag_profile", { city }),
  history: (ward_id, horizon) => get("/api/history", { ward_id, horizon }),
  whatIf: (body) => post("/api/what_if", body),
  // Phase 2: allocation, overflow, crew simulator, dispatch
  allocate: (body) => post("/api/allocate", body),           // {as_of, horizon, stock_pct, crews?, demand_basis?, city?}
  overflow: (as_of, horizon, beds_available_pct, demand_basis) => get("/api/overflow", { as_of, horizon, beds_available_pct, demand_basis }),
  roi: (as_of, crews_per_ward) => get("/api/intervention/roi", { as_of, crews_per_ward }),
  intervention: (body) => post("/api/intervention", body),   // {as_of, deployment: {ward_id: crews}, total_crews?}
  crewPlan: (as_of, crews) => get("/api/crew_plan", { as_of, crews }),
  dispatch: (q) => get("/api/dispatch", q),                  // q = {as_of, horizon, stock_pct, crews, beds_available_pct, lang, city}
  briefUrl: (q) => `/api/dispatch/brief?${clean(q)}`,       // a page to open, not JSON to fetch
  supplyPlan: (as_of, city) => get("/api/supply_plan", { as_of, city }),
};

/** Query string without empty values (unset settings fall back to the server's defaults). */
function clean(params) {
  return new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ""));
}
