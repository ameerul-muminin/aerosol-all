// Cost & speed: measured costs from reports/efficiency.json (scripts/measure_efficiency.py) and live cache
// counters of the running server, both via GET /api/efficiency; plus this browser's own reuse count.
import { api, browserCache } from "../api.js";
import { t } from "../i18n.js";
import { num0, num1, esc } from "../format.js";
import { showError, ticket } from "../ui.js";

const pct = (x) => `${num0(x * 100)}%`;
/** 0.0006 s -> "0.6 ms", 0.76 s -> "760 ms", 99.9 s -> "100 s" */
const dur = (sec) => (sec < 0.01 ? t("cost.ms", { n: num1(sec * 1000) })
  : sec < 1 ? t("cost.ms", { n: num0(sec * 1000) }) : t("cost.s", { n: num0(sec) }));
const kb = (bytes) => t("cost.kb", { n: num1(bytes / 1000) });
// measured timings shown in the list, in this order (a key missing from the file is skipped)
const TIMINGS = ["app_startup_warm", "model_load_joblib", "lag_features_build_all_wards",
  "forecast_one_ward_cached_median", "forecast_one_ward_uncached_median", "ranking_60_wards_cached_median",
  "ranking_60_wards_first_request", "warnings_60_wards_cached_median", "warnings_60_wards_first_request",
  "lag_profile_cached_median", "lag_profile_first_request", "what_if_repeat_scenario_median",
  "what_if_new_scenario", "full_retrain_3_horizons"];

export async function render(root, { retry }) {
  const current = ticket(root);
  if (!root.querySelector(".cost")) {
    root.innerHTML = `<div class="head"><h1 class="display">${esc(t("cost.title"))}</h1><p><span class="skel">${esc(t("state.loading"))}</span></p></div>
      <div class="grid" aria-busy="true"><section class="card span-12"><span class="skel" style="height:320px;width:100%"></span></section></div>`;
  }
  let e;
  try {
    e = await api.efficiency();
  } catch (err) {
    if (current()) showError(root, err, retry);
    return;
  }
  if (!current()) return;
  const m = e.measured;
  if (!m) { root.innerHTML = `<div class="head cost"><h1 class="display">${esc(t("cost.title"))}</h1></div><div class="state"><p>${esc(t("cost.not_measured"))}</p></div>`; return; }
  const sec = m.seconds || {};
  const by = m.bytes || {};

  root.innerHTML = `
  <div class="head cost">
    <h1 class="display">${esc(t("cost.title"))}</h1>
    <p>${esc(t("cost.subtitle"))}</p>
  </div>
  <div class="grid">
    ${tile(t("cost.t_model"), t("cost.mb", { n: num1(m.model_file_mb) }), t("cost.t_model_note"))}
    ${tile(t("cost.t_ward"), dur(sec.forecast_one_ward_cached_median), t("cost.t_ward_note", { fresh: dur(sec.forecast_one_ward_uncached_median) }))}
    ${tile(t("cost.t_all"), dur(sec.ranking_60_wards_cached_median), t("cost.t_all_note", { fresh: dur(sec.ranking_60_wards_first_request) }))}
    ${tile(t("cost.t_ext"), num0((m.external_api_calls || 0) + (m.llm_calls || 0)), t("cost.t_ext_note"))}
    <section class="card span-7" id="live">${live(e.live)}</section>
    <section class="card span-5"><h2>${esc(t("cost.size_title"))}</h2>${size(by)}</section>
    <section class="card span-12"><h2>${esc(t("cost.timing_title"))}</h2>
      <dl class="dl cols">${TIMINGS.filter((k) => sec[k] != null).map((k) => `<div><dt>${esc(t(`cost.k_${k}`))}</dt><dd>${esc(dur(sec[k]))}</dd></div>`).join("")}</dl>
      <p class="acc-note">${esc(t("cost.measured", { when: m.measured_at, where: m.platform, py: m.python, n: num0(m.calls_per_median || 0) }))}</p>
    </section>
  </div>`;

  root.querySelector("#live").addEventListener("click", async (ev) => {
    if (!ev.target.closest("#live-refresh")) return;
    try { const fresh = await api.efficiency(); root.querySelector("#live").innerHTML = live(fresh.live); } catch { /* keep the old numbers */ }
  });
}

function tile(label, value, note) {
  return `<section class="card tile span-3"><span class="k">${esc(label)}</span><b>${esc(value)}</b><span class="of">${esc(note)}</span></section>`;
}

/** Three cache layers, each "reused vs computed" with its own scope: the model's all-ward results and the
 *  server's answers (since the server started, every user), and this browser tab's own memory. */
function live(l) {
  const layers = [
    ["model", l.model_results_reused, l.model_results_needed - l.model_results_reused],
    ["server", l.answers_from_cache, l.answers_computed],
    ["tab", browserCache.reused, browserCache.asked],
  ];
  const rows = layers.map(([k, hit, miss]) => {
    const n = hit + miss;
    return `<div class="layer"><div class="top"><span>${esc(t(`cost.l_${k}`))}</span>
        <b>${esc(t(k === "tab" ? "cost.l_counts_tab" : "cost.l_counts", { hit: num0(hit), miss: num0(miss) }))}</b></div>
      <span class="split" role="img" aria-label="${esc(`${t(`cost.l_${k}`)}: ${t(k === "tab" ? "cost.l_counts_tab" : "cost.l_counts", { hit: num0(hit), miss: num0(miss) })}`)}">
        ${n ? `<i class="hit" style="width:${(hit / n) * 100}%"></i><i class="miss" style="width:${(miss / n) * 100}%"></i>` : ""}</span></div>`;
  }).join("");
  const share = l.model_results_needed ? l.model_results_reused / l.model_results_needed : null;
  return `<div class="clk-head"><h2>${esc(t("cost.live_title"))}</h2>
      <button type="button" class="btn small" id="live-refresh">${esc(t("cost.refresh"))}</button></div>
    <p class="live-big"><b>${share == null ? "–" : esc(pct(share))}</b><span>${esc(t("cost.live_sum", { runs: num0(l.model_results_needed - l.model_results_reused), n: num0(l.model_results_needed) }))}</span></p>
    <div class="legend"><span><i style="background:var(--tier-low)"></i>${esc(t("cost.lg_hit"))}</span><span><i style="background:var(--bar)"></i>${esc(t("cost.lg_miss"))}</span></div>
    <div class="layers">${rows}</div>
    <p class="acc-note">${esc(t("cost.live_note"))}</p>`;
}

function size(by) {
  if (!by.ranking_response_raw) return "";
  const r = by.ranking_response_gzip / by.ranking_response_raw;
  return `<div class="gz">
      <div><span>${esc(t("cost.raw"))}</span><i style="width:100%"></i><b>${esc(kb(by.ranking_response_raw))}</b></div>
      <div><span>${esc(t("cost.gzip"))}</span><i class="z" style="width:${r * 100}%"></i><b>${esc(kb(by.ranking_response_gzip))}</b></div>
    </div>
    <p class="clk-sum" style="margin:16px 0 0">${esc(t("cost.size_sum", { p: pct(1 - r) }))}</p>
    <p class="acc-note">${esc(t("cost.size_note"))}</p>`;
}
