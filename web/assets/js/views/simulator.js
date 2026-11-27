// V4 Crew simulator: ROI ranking (cases prevented per crew), build a deployment with steppers, run the
// model-driven simulation (crews lower each ward's water-logging input, the forecast models re-run at +2/+3/+4),
// or fill the plan from the best placement. Data: GET /api/intervention/roi, POST /api/intervention, GET /api/crew_plan.
// The result always matches the plan on screen: while a change is being run the old result is dimmed, and an
// empty or over-budget plan shows why there is no result instead of an old one.
import { api } from "../api.js";
import { t } from "../i18n.js";
import { num0, num1, num2, dateLong, cityName, wardName, esc } from "../format.js";
import { showError, ticket, segment, debounce, loadingHTML, inlineError, downloadCsv } from "../ui.js";

// the deployment survives language switches and visits to other tabs, but not a new "As of" week
let dep = new Map();
let depAsOf = "";
let cpw = 1;            // crews per ward used for the ranking
let suggestMsg = "";
let rankAll = false;     // ranking shows the top 20 until "Show all"
const SHOWN = 20;

export async function render(root, { state, save, retry }) {
  const s = { ...state };
  if (depAsOf !== s.as_of) { dep = new Map(); depAsOf = s.as_of; suggestMsg = ""; }
  const cur = ticket(root);
  root.innerHTML = loadingHTML(t("sim.title"));
  let roi, plan;
  try {
    [roi, plan] = await Promise.all([api.roi(s.as_of, cpw), api.crewPlan(s.as_of, s.crews)]);
  } catch (err) {
    if (cur()) showError(root, err, retry);
    return;
  }
  if (!cur()) return;

  const maxPer = roi.assumptions.crews.max_per_ward.value;
  const placeable = roi.wards.length * maxPer;    // no plan can use more crews than wards x per-ward cap
  if (cpw > maxPer) cpw = 1;
  let total = Math.min(s.crews !== "" ? Number(s.crews) : plan.crews_available, placeable);
  if (s.crews !== "" && Number(s.crews) !== total) save({ crews: String(total) });   // a shared link asked for more
  for (const [id, n] of dep) if (n > maxPer) dep.set(id, maxPer);
  let lastEffect = roi.assumptions.crew_effect;   // replaced by the run answer's `assumption` after a run
  let lastRun = null;
  let lastKey = "";                               // the plan + total that lastRun answers
  let running = 0;                                // simulation requests in flight
  let failed = false;                             // the last run answered with an error

  root.innerHTML = `
  <div class="head"><h1 class="display">${esc(t("sim.title"))}</h1><p>${esc(t("sim.subtitle", { week: dateLong(s.as_of) }))}</p></div>
  <div class="toolbar">
    <div class="field"><label for="v-total">${esc(t("sim.total"))}</label>
      <input type="number" id="v-total" min="0" max="${placeable}" step="1" inputmode="numeric" value="${total}" aria-describedby="v-total-hint">
      <span class="field-hint" id="v-total-hint" aria-live="polite"></span></div>
    <div class="field"><span id="vc-l">${esc(t("sim.per_ward"))}</span>
      <div class="seg" id="v-cpw" role="group" aria-labelledby="vc-l"><span class="thumb"></span>
        ${Array.from({ length: maxPer }, (_, i) => `<button type="button" data-v="${i + 1}">${num0(i + 1)}</button>`).join("")}</div></div>
  </div>
  <div class="grid sim-top">
    <section class="card span-7" id="v-rank"></section>
    <section class="card span-5 sticky" id="v-dep"></section>
  </div>
  <div class="grid"><section class="card span-12 out" id="v-res" aria-live="polite"></section></div>`;

  const rankEl = root.querySelector("#v-rank");
  const depEl = root.querySelector("#v-dep");
  const resEl = root.querySelector("#v-res");
  const totalEl = root.querySelector("#v-total");
  const hintEl = root.querySelector("#v-total-hint");

  const used = () => [...dep.values()].reduce((a, b) => a + b, 0);
  const key = () => JSON.stringify([[...dep.entries()].sort(), total]);
  const status = () => (!used() ? "empty" : used() > total ? "over" : "ok");

  function paintRank() {
    const rows = (rankAll ? roi.wards : roi.wards.slice(0, SHOWN)).map((w) => `<tr>
      <td class="r">${num0(w.rank)}</td>
      <td><b>${esc(wardName(w.ward_id))}</b><span class="city">${esc(cityName(w.city))}</span></td>
      <td>${stepper(w.ward_id)}</td>
      <td class="num">${num1(w.cases_prevented)}</td>
      <td class="num c">${num2(w.roi)}<span class="sub-line">${num2(w.roi_low)}–${num2(w.roi_high)}</span></td>
      <td>${w.floor_hit ? `<span class="badge">${esc(t("sim.at_limit"))}</span>` : ""}</td></tr>`).join("");
    const n = roi.crews_per_ward;
    rankEl.innerHTML = `<div class="warn-head"><h2>${esc(t("sim.rank_title"))}</h2><span>${esc(t(n === 1 ? "sim.rank_note_one" : "sim.rank_note", { n: num0(n) }))}</span></div>
    <div class="table-wrap"><table class="rank">
      <thead><tr><th scope="col">${esc(t("wards.rank"))}</th><th scope="col">${esc(t("wards.ward"))}</th><th scope="col">${esc(t("sim.col_crews"))}</th>
        <th scope="col" class="num">${esc(t("sim.col_prevented"))}</th><th scope="col" class="num">${esc(t("sim.col_roi"))}</th>
        <th scope="col"><span class="sr-only">${esc(t("sim.at_limit"))}</span></th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <div class="more-row"><button type="button" class="btn small" id="v-all">${esc(rankAll ? t("alloc.show_less") : t("alloc.show_all", { n: num0(roi.wards.length) }))}</button></div>`;
    rankEl.querySelector("#v-all").onclick = () => { rankAll = !rankAll; paintRank(); rankEl.querySelector("#v-all").focus(); };
  }

  function stepper(id) {
    const n = dep.get(id) || 0;
    const name = wardName(id);
    const full = n >= maxPer || used() >= total;   // the ward's cap, or every available crew is placed
    return `<span class="stp"><button type="button" data-step="-1" data-ward="${id}" aria-label="${esc(t("sim.remove", { ward: name }))}"${n <= 0 ? " disabled" : ""}>−</button>
      <output data-ward="${id}">${num0(n)}</output>
      <button type="button" data-step="1" data-ward="${id}" aria-label="${esc(t("sim.add", { ward: name }))}"${full ? " disabled" : ""}>+</button></span>`;
  }

  function paintDep() {
    const n = used();
    const st = status();
    const items = [...dep.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    const note = st === "over" ? t("sim.over", { n: num0(n), total: num0(total) })
      : st === "ok" && n >= total ? t("sim.full", { total: num0(total) }) : "";
    depEl.innerHTML = `<h2>${esc(t("sim.dep_title"))}</h2>
      <p class="dep-used${st === "over" ? " deficit" : ""}" aria-live="polite">${esc(t("sim.used", { n: num0(n), total: num0(total) }))}</p>
      ${note ? `<p class="dep-note${st === "over" ? " deficit" : ""}">${esc(note)}</p>` : ""}
      <div class="dep-row">
        <button type="button" class="btn primary" id="v-suggest">${esc(t("sim.suggest"))}</button>
        <button type="button" class="btn" id="v-run"${st !== "ok" ? " disabled" : ""}>${esc(t("sim.run"))}</button>
        ${n ? `<button type="button" class="btn small" id="v-clear">${esc(t("sim.clear"))}</button>` : ""}
      </div>
      <p class="inline-error" role="alert" id="v-err" hidden></p>
      ${suggestMsg ? `<p class="dep-msg">${esc(suggestMsg)}</p>` : ""}
      ${items.length ? `<ul class="dep">${items.map(([id]) => {
        const w = roi.wards.find((x) => x.ward_id === id);
        return `<li><span><b>${esc(wardName(id))}</b> <span class="city">${esc(cityName(w ? w.city : ""))}</span></span>${stepper(id)}</li>`;
      }).join("")}</ul>` : `<p class="dep-empty">${esc(t("sim.dep_empty"))}</p>`}`;
    depEl.querySelector("#v-run").onclick = run;
    depEl.querySelector("#v-suggest").onclick = suggest;
    const clear = depEl.querySelector("#v-clear");
    if (clear) clear.onclick = () => { dep.clear(); suggestMsg = ""; changed(); depEl.querySelector("#v-suggest").focus(); };
  }

  /** Repaint steppers in the ranking (in place) and the panel, keeping keyboard focus on the same button. */
  function refresh() {
    const a = document.activeElement;
    const keep = a && a.dataset && a.dataset.step ? `[data-step="${a.dataset.step}"][data-ward="${a.dataset.ward}"]` : null;
    const inPanel = keep && depEl.contains(a);
    rankEl.querySelectorAll(".stp").forEach((el) => {
      const id = el.querySelector("output").dataset.ward;
      el.outerHTML = stepper(id);
    });
    paintDep();
    if (keep) {
      const target = (inPanel ? depEl : rankEl).querySelector(keep);
      const live = target && (target.disabled ? target.parentElement.querySelector("button:not([disabled])") : target);
      (live || depEl.querySelector("#v-suggest")).focus();   // the ward left the panel, or both buttons are off
    }
  }

  /** The plan or the total changed: repaint, drop a result that no longer fits, re-run after a short pause. */
  function changed() {
    refresh();
    paintRes();
    if (status() === "ok") autoRun();
  }

  root.querySelector(".sim-top").addEventListener("click", (e) => {   // rebuilt on each render: no stacked listeners
    const b = e.target.closest("[data-step]");
    if (!b || b.disabled) return;
    const id = b.dataset.ward;
    const n = Math.max(0, Math.min(maxPer, (dep.get(id) || 0) + Number(b.dataset.step)));
    if (n) dep.set(id, n); else dep.delete(id);
    suggestMsg = "";
    changed();
  });

  async function run() {
    if (status() !== "ok") return;
    const mine = ticket(resEl);
    const k = key();
    running++;
    paintRes();
    try {
      const r = await api.intervention({ as_of: s.as_of, deployment: Object.fromEntries(dep), total_crews: total });
      if (!mine() || !resEl.isConnected) return;
      inlineError(depEl.querySelector("#v-err"), null);
      lastRun = r;
      lastKey = k;
      lastEffect = r.assumption;
      failed = false;
    } catch (e) {
      if (mine()) { failed = true; inlineError(depEl.querySelector("#v-err"), e); }
    } finally {
      running--;
      if (resEl.isConnected) paintRes();
    }
  }
  const autoRun = debounce(() => { if (status() === "ok" && key() !== lastKey) run(); }, 400);

  async function suggest() {
    const btn = depEl.querySelector("#v-suggest");
    btn.disabled = true;
    const mine = ticket(depEl);
    const asked = total;
    try {
      const p = await api.crewPlan(s.as_of, asked);
      if (!mine() || !depEl.isConnected || total !== asked) return;   // the total changed while we waited
      dep = new Map(Object.entries(p.plan));
      suggestMsg = t("sim.suggest_msg", { n: num1(p.cases_prevented), m: num1(p.compare.allocation_rule.cases_prevented) });
      if (p.crews_used < asked) suggestMsg += ` ${t("sim.suggest_spare", { n: num0(p.crews_used), total: num0(asked) })}`;
      changed();
      depEl.querySelector("#v-suggest").focus();
    } catch (e) {
      if (!mine()) return;
      inlineError(depEl.querySelector("#v-err"), e);
      btn.disabled = false;
    }
  }

  function paintRes() {
    const honest = honesty(lastEffect);
    const st = status();
    // dimmed while a run is out, or while the shown answer is for an older plan (the new run is on its way)
    const busy = st === "ok" && (running > 0 || (lastRun && lastKey !== key()));
    if (busy) resEl.setAttribute("aria-busy", "true"); else resEl.removeAttribute("aria-busy");
    const msg = st === "empty" ? t("sim.res_empty") : st === "over" ? t("sim.res_over")
      : !lastRun ? t(failed ? "sim.res_empty" : "state.loading") : "";   // a valid plan is always about to run
    if (msg) {
      resEl.innerHTML = `<h2>${esc(t("sim.res_title"))}</h2><p class="warn-none">${esc(msg)}</p><p class="honest">${esc(honest)}</p>`;
      return;
    }
    const r = lastRun;
    const cards = r.horizons.map((h) => `<div>
      <span class="k">${esc(t("outlook.plus_weeks", { h: num0(h.horizon) }))} · ${esc(dateLong(h.forecast_week))}</span>
      <b>${num1(h.cases_before)} → ${num1(h.cases_after)}</b>
      <p><strong>${esc(t("sim.prevented", { n: num1(h.cases_prevented) }))}</strong> ${esc(t("sim.range", { low: num1(h.cases_prevented_low), high: num1(h.cases_prevented_high) }))}</p>
      <div class="ba" role="img" aria-label="${esc(t("sim.ba_aria", { a: num1(h.cases_before), b: num1(h.cases_after) }))}">
        <span>${esc(t("sim.before"))}</span><span class="hbar"><i style="transform:scaleX(1)"></i></span>
        <span>${esc(t("sim.after"))}</span><span class="hbar"><i class="after" style="transform:scaleX(${h.cases_before > 0 ? h.cases_after / h.cases_before : 0})"></i></span></div>
      <p>${esc(t("sim.beds_freed", { n: num0(h.beds_freed), b: num0(h.beds_before), a: num0(h.beds_after) }))}</p>
      <p>${esc(t("sim.saline_saved", { n: num0(h.saline_saved), b: num0(h.saline_before), a: num0(h.saline_after) }))}</p>
    </div>`).join("");
    const rows = r.wards.map((w) => `<tr><td><b>${esc(wardName(w.ward_id))}</b><span class="city">${esc(cityName(w.city))}</span></td>
      <td class="num">${num0(w.crews)}</td>
      ${["2", "3", "4"].map((k) => `<td class="num">${num2(w.by_horizon[k].cases_prevented)}</td>`).join("")}
      <td class="num c">${num2(w.roi)}<span class="sub-line">${num2(w.roi_low)}–${num2(w.roi_high)}</span></td>
      <td>${w.floor_hit ? `<span class="badge">${esc(t("sim.at_limit"))}</span>` : ""}</td></tr>`).join("");
    resEl.innerHTML = `<div class="warn-head"><h2>${esc(t("sim.res_title"))}</h2><span>${esc(t("sim.res_scope", { w: num0(roi.wards.length), n: num0(r.crews_used), total: num0(r.total_crews) }))}</span></div>
      <div class="simh">${cards}</div>
      <div class="table-wrap"><table class="rank">
        <thead><tr><th scope="col">${esc(t("wards.ward"))}</th><th scope="col" class="num">${esc(t("sim.col_crews"))}</th>
          ${["2", "3", "4"].map((k) => `<th scope="col" class="num">${esc(t("sim.col_prev_h", { h: num0(Number(k)) }))}</th>`).join("")}
          <th scope="col" class="num">${esc(t("sim.col_roi"))}</th><th scope="col"><span class="sr-only">${esc(t("sim.at_limit"))}</span></th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <div class="more-row"><button type="button" class="btn small" id="v-csv">${esc(t("wards.csv"))}</button></div>
      <p class="honest">${esc(honest)}</p>`;
    resEl.querySelector("#v-csv").onclick = () => csv(r);
  }

  paintRank();
  paintDep();
  paintRes();
  if (status() === "ok") run();

  let cpwShown = cpw;
  const pickCpw = async (v) => {
    const mine = ticket(rankEl);
    cpw = Number(v);
    rankEl.setAttribute("aria-busy", "true");
    try {
      const next = await api.roi(s.as_of, cpw);
      if (!mine() || !rankEl.isConnected) return;     // a later click won: its answer is the one to show
      roi = next;
      cpwShown = cpw;
      paintRank();
    } catch (e) {
      if (!mine()) return;
      cpw = cpwShown;                                  // put the control back on what the table shows
      segment(root.querySelector("#v-cpw"), cpw, pickCpw);
      rankEl.querySelector(".rank-err")?.remove();
      rankEl.insertAdjacentHTML("afterbegin", `<p class="inline-error rank-err" role="alert">${esc(e.message)}</p>`);
    } finally {
      if (mine()) rankEl.removeAttribute("aria-busy");
    }
  };
  segment(root.querySelector("#v-cpw"), cpw, pickCpw);

  const saveTotal = debounce((v) => { save({ crews: v }); }, 200);
  totalEl.oninput = () => {
    const v = totalEl.value.trim();
    if (!/^\d+$/.test(v)) { hintEl.textContent = t("sim.total_bad", { max: num0(placeable) }); return; }
    const n = Math.min(Number(v), placeable);
    hintEl.textContent = Number(v) > placeable ? t("sim.total_cap", { max: num0(placeable), w: num0(roi.wards.length), k: num0(maxPer) }) : "";
    if (n === total) return;
    total = n;
    s.crews = String(n);
    suggestMsg = "";
    saveTotal(String(n));
    changed();
  };
  totalEl.onchange = () => {                           // on leaving the field, show the number actually used
    const capped = /^\d+$/.test(totalEl.value.trim()) && Number(totalEl.value) > placeable;
    totalEl.value = String(total);
    if (!capped) hintEl.textContent = "";              // keep only the "at most N can be placed" note
  };
}

/** The required honesty line: every number is read from the API's crew-effect assumption. */
function honesty(a) {
  const val = (x) => (x && typeof x === "object" && "value" in x ? x.value : x);
  const pct = (x) => `${num0(val(x) * 100)}%`;
  const sens = (val(a.sensitivity_per_crew) || []).map((x) => pct(x)).join(" / ");
  return t("sim.honest", { r: pct(a.reduction_per_crew), max: pct(a.max_total_reduction), sens });
}

/** The simulated plan as a spreadsheet: one row per ward, cases prevented per week, read from the run answer. */
function csv(r) {
  const head = ["ward_id", "city", "crews", "prevented_h2", "prevented_h3", "prevented_h4", "cases_prevented",
    "cases_prevented_low", "cases_prevented_high", "roi", "roi_low", "roi_high", "floor_hit"];
  const rows = r.wards.map((w) => [w.ward_id, w.city, w.crews, ...["2", "3", "4"].map((k) => w.by_horizon[k].cases_prevented),
    w.cases_prevented, w.cases_prevented_low, w.cases_prevented_high, w.roi, w.roi_low, w.roi_high, w.floor_hit]);
  downloadCsv(`dengue_crew_plan_${r.as_of}_${r.crews_used}crews.csv`, head, rows);
}
