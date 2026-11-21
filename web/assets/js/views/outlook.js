// F2 Ward outlook + F3 what-if panel. Data: GET /api/forecast, POST /api/what_if.
import { api } from "../api.js";
import { t } from "../i18n.js";
import {
  num0, num1, num2, signed1, dateLong, dateShort, cityName, wardName, wardFull,
  tierHTML, tierName, driverName, driverDetail, esc,
} from "../format.js";
import { showError, ticket } from "../ui.js";

const wiValues = { rain: "", temp: "", hum: "", weeks: "4" };   // what-if inputs survive re-renders

function skeleton(root, s) {
  root.innerHTML = `
  <div class="head"><h1 class="display"><span class="skel">${esc(wardName(s.ward))}</span></h1><p><span class="skel">${esc(t("state.loading"))}</span></p></div>
  <div class="grid" aria-busy="true">
    <section class="card span-6"><span class="skel" style="height:120px;width:60%"></span></section>
    <section class="card span-6"><span class="skel" style="height:120px;width:100%"></span></section>
    <section class="card span-12"><span class="skel" style="height:200px;width:100%"></span></section>
  </div>`;
}

export async function render(root, { state: s, set, retry }) {
  const current = ticket(root);
  if (!root.querySelector(".verdict")) skeleton(root, s);
  let f;
  try {
    f = await api.forecast(s.ward, s.as_of, s.h);
  } catch (err) {
    if (current()) showError(root, err, retry);
    return;
  }
  if (!current()) return;

  root.innerHTML = `
  <div class="head">
    <h1 class="display">${esc(wardName(f.ward_id))}</h1>
    <p>${esc(t("outlook.subtitle", { city: cityName(f.city), week: dateLong(f.forecast_week) }))}</p>
  </div>
  <div class="grid">
    <section class="card verdict span-6" aria-live="polite">${verdict(f)}</section>
    <section class="card span-6">
      <h2 id="hz-h">${esc(t("outlook.horizon_title"))}</h2>
      <div class="hzrow" role="group" aria-labelledby="hz-h">${horizons(f, s.h)}</div>
    </section>
    <section class="card span-12">
      <h2>${esc(t("outlook.rain_title"))}</h2>
      <div class="legend">
        <span><i style="background:var(--bar)"></i>${esc(t("outlook.rain_legend_rain"))}</span>
        <span><i style="background:var(--accent)"></i>${esc(t("outlook.rain_legend_window"))}</span>
        <span><i style="box-shadow:inset 0 0 0 1px var(--line-strong)"></i>${esc(t("outlook.rain_legend_unknown"))}</span>
        <span><i style="box-shadow:inset 0 0 0 1.5px var(--ink)"></i>${esc(t("outlook.rain_legend_forecast"))}</span>
      </div>
      ${rain(f)}
    </section>
    <section class="card span-7">
      <h2>${esc(t("outlook.why_title"))}</h2>
      ${why(f)}
    </section>
    <section class="card span-5">
      <h2>${esc(t("outlook.inputs_title"))}</h2>
      ${inputs(f)}
    </section>
    <section class="card span-12" id="whatif">${whatIfForm(f)}</section>
  </div>`;

  root.querySelectorAll(".hzrow button").forEach((b) => (b.onclick = () => set({ h: Number(b.dataset.h) })));
  wireWhatIf(root.querySelector("#whatif"), f);
}

function verdict(f) {
  const max = Math.ceil(Math.max(20, f.range_high * 1.2) / 5) * 5;
  const p = (v) => (Math.min(v, max) / max) * 100;
  return `
    <div class="lbl">${esc(t("outlook.expected"))}</div>
    <div class="n">${num1(f.predicted_cases)}</div>
    <div>
      <div class="tierline">${tierHTML(f.risk_tier, { risk: true })}</div>
      <p class="rng">${esc(t("outlook.range", { low: num0(f.range_low), high: num0(f.range_high) }))}</p>
      <div class="rbar" aria-hidden="true"><i style="left:${p(f.range_low)}%;width:${p(f.range_high) - p(f.range_low)}%"></i><b style="left:calc(${p(f.predicted_cases)}% - 1px)"></b></div>
      <div class="rscale" aria-hidden="true"><span>${num0(0)}</span><span>${num0(max)}</span></div>
    </div>
    <p class="act">${esc(t(`action.${f.risk_tier}`))}</p>`;
}

function horizons(f, h) {
  return Object.keys(f.by_horizon).map((k) => {
    const b = f.by_horizon[k];
    return `<button type="button" data-h="${k}" aria-pressed="${Number(k) === h}">
      <span class="w">${esc(t("outlook.plus_weeks", { h: num0(Number(k)) }))}</span>
      <b>${num1(b.predicted_cases)}</b>${tierHTML(b.risk_tier)}</button>`;
  }).join("");
}

function rain(f) {
  const tl = f.rain_timeline;
  const max = Math.max(1, ...tl.map((d) => d.rain_mm || 0));
  const known = tl.filter((d) => d.rain_mm != null).map((d) => `${dateShort(d.week)} ${num0(d.rain_mm)} ${t("outlook.mm")}`).join(", ");
  const cols = tl.map((d, i) => {
    const cls = d.status === "forecast" ? "f" : d.status === "unknown" ? "u" : d.in_breeding_window ? "bw" : "";
    const label = d.rain_mm == null ? "" : num0(d.rain_mm);
    const h = d.rain_mm == null ? 100 : Math.max(2, (d.rain_mm / max) * 100);
    return `<div class="c ${cls}"><span class="mm">${label || "&nbsp;"}</span><i style="height:${h}%;animation-delay:${i * 16}ms"></i><em>${esc(dateShort(d.week))}</em></div>`;
  }).join("");
  return `<div class="rain" style="--n:${tl.length}" role="img" aria-label="${esc(t("outlook.rain_aria", { list: known }))}">${cols}</div>`;
}

function why(f) {
  const ref = f.driver_reference_cases;
  const ds = [...f.top_drivers];
  const rest = f.predicted_cases - ref - ds.reduce((a, d) => a + d.effect, 0);
  if (Math.abs(rest) >= 0.05) ds.push({ label: "other", effect: rest });
  let run = ref, peak = Math.max(ref, f.predicted_cases);
  ds.forEach((d) => { run += d.effect; peak = Math.max(peak, run); });
  const P = (v) => (Math.max(0, v) / (peak * 1.04)) * 100;
  run = ref;
  const rows = [`<div class="lab end">${esc(t("outlook.why_avg"))}</div><div class="val">${num1(ref)}</div>
    <div class="track"><span class="s2" style="left:0;width:${P(ref)}%;background:var(--bar-2)"></span></div>`];
  ds.forEach((d, i) => {
    const a = run, b = run + d.effect; run = b;
    const cls = d.effect >= 0 ? "up" : "down";
    const name = driverName(d.label), detail = driverDetail(d.label);
    rows.push(`<div class="lab"${detail ? ` title="${esc(detail)}"` : ""}>${esc(name)}</div><div class="val ${cls}">${signed1(d.effect)}</div>
      <div class="track" aria-hidden="true"><span class="s2" style="left:${P(Math.min(a, b))}%;width:${Math.max(0.6, Math.abs(P(b) - P(a)))}%;background:var(--${d.effect >= 0 ? "raises" : "lowers"});animation-delay:${(i + 1) * 30}ms"></span></div>`);
  });
  rows.push(`<hr><div class="lab end">${esc(t("outlook.why_this"))}</div><div class="val">${num1(f.predicted_cases)}</div>
    <div class="track"><span class="s2" style="left:0;width:${P(f.predicted_cases)}%;background:var(--ink)"></span></div>`);
  return `<p class="sum">${t("outlook.why_sum", { ref: `<b>${num1(ref)}</b>`, pred: `<b>${num1(f.predicted_cases)}</b>` })}</p>
    <div class="wf">${rows.join("")}</div>`;
}

function inputs(f) {
  const r = f.recent, w = f.ward_profile;
  const rows = [
    ["outlook.in_rain4", `${num0(r.rain_last_4_weeks_mm)} ${t("outlook.mm")}`],
    ["outlook.in_humidity", `${num0(r.humidity_pct)}%`],
    ["outlook.in_temp", `${num1(r.avg_temp_c)} °C`],
    ["outlook.in_cases", num0(r.cases_this_week)],
    ["outlook.in_waterlog", num2(w.water_logging_risk_index)],
    ["outlook.in_sanitation", num2(w.sanitation_score)],
    ["outlook.in_density", num0(w.population_density_sqkm)],
  ];
  return `<dl class="dl">${rows.map(([k, v]) => `<div><dt>${esc(t(k))}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>`;
}

// ---- F3 what-if ---------------------------------------------------------------------
function whatIfForm(f) {
  const r = f.recent;
  const ph = { rain: num0(r.rain_last_4_weeks_mm / 4), temp: num1(r.avg_temp_c), hum: num0(r.humidity_pct) };
  const field = (id, key, unit, attrs) => `<label class="field" for="wi-${id}">
      <span>${esc(t(`whatif.${key}`))} <span class="unit">(${esc(t(`whatif.${unit}`))})</span></span>
      <input type="number" id="wi-${id}" name="${id}" inputmode="decimal" placeholder="${esc(ph[id])}" value="${esc(wiValues[id])}" ${attrs}></label>`;
  const weeks = Array.from({ length: 8 }, (_, i) => i + 1)
    .map((n) => `<option value="${n}"${String(n) === wiValues.weeks ? " selected" : ""}>${esc(t("whatif.weeks_n", { n: num0(n) }))}</option>`).join("");
  return `<h2>${esc(t("whatif.title"))}</h2>
  <form class="wi-form" novalidate>
    ${field("rain", "rain", "rain_unit", 'min="0" max="400" step="1"')}
    ${field("temp", "temp", "temp_unit", 'step="0.1"')}
    ${field("hum", "hum", "hum_unit", 'min="0" max="100" step="1"')}
    <label class="field" for="wi-weeks"><span>${esc(t("whatif.weeks"))}</span><select id="wi-weeks" name="weeks">${weeks}</select></label>
    <div class="wi-row">
      <span class="hint">${esc(t("whatif.blank_hint"))}</span>
      <button type="button" class="btn small" data-preset="0">${esc(t("whatif.preset_dry"))}</button>
      <button type="button" class="btn small" data-preset="250">${esc(t("whatif.preset_wet"))}</button>
      <button type="submit" class="btn primary">${esc(t("whatif.run"))}</button>
    </div>
  </form>
  <div class="wi-out" aria-live="polite"></div>`;
}

function wireWhatIf(card, f) {
  const form = card.querySelector("form");
  const out = card.querySelector(".wi-out");
  const submit = form.querySelector('[type="submit"]');
  form.addEventListener("input", (e) => { if (e.target.name in wiValues) wiValues[e.target.name] = e.target.value; });
  form.querySelectorAll("[data-preset]").forEach((b) => (b.onclick = () => {
    form.rain.value = wiValues.rain = b.dataset.preset;
    form.requestSubmit();
  }));
  form.onsubmit = async (e) => {
    e.preventDefault();
    const body = { ward_id: f.ward_id, as_of: f.as_of, horizon: f.horizon, weeks: Number(form.weeks.value) };
    const map = { rain: "rainfall_mm", temp: "avg_temp_c", hum: "relative_humidity_pct" };
    let any = false;
    for (const [id, key] of Object.entries(map)) {
      const v = form[id].value.trim();
      if (v !== "") { body[key] = Number(v); any = true; }
    }
    if (!any) { out.innerHTML = `<p class="inline-error" role="alert">${esc(t("whatif.need_one"))}</p>`; return; }
    submit.disabled = true;
    submit.textContent = t("whatif.running");
    try {
      out.innerHTML = result(await api.whatIf(body));
    } catch (err) {
      out.innerHTML = `<p class="inline-error" role="alert">${esc(err.message)}</p>`;
    } finally {
      submit.disabled = false;
      submit.textContent = t("whatif.run");
    }
  };
}

function result(w) {
  const box = (key, x) => `<div><span class="k">${esc(t(key))}</span><b>${num1(x.predicted_cases)}</b>${tierHTML(x.risk_tier)}
      <p class="r">${esc(t("outlook.range", { low: num0(x.range_low), high: num0(x.range_high) }))}</p></div>`;
  const c = w.change.cases;
  const tierLine = w.change.tier_changed
    ? t("whatif.tier_changed", { from: tierName(w.baseline.risk_tier), to: tierName(w.scenario.risk_tier) })
    : t("whatif.tier_same", { tier: tierName(w.scenario.risk_tier) });
  const a = w.applied || {};
  const rainMean = a.replaced_actual && a.replaced_actual.rainfall_mm ? a.replaced_actual.rainfall_mm.mean : null;
  const note = a.weeks_replaced && rainMean != null
    ? `<p class="wi-note">${esc(t("whatif.replaced", { from: dateShort(a.weeks_replaced.from), to: dateShort(a.weeks_replaced.to), mean: num0(rainMean) }))}</p>` : "";
  const warns = (w.warnings || []).length ? `<ul class="wi-warn">${w.warnings.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  return `<div class="wi-cmp">${box("whatif.baseline", w.baseline)}${box("whatif.scenario", w.scenario)}</div>
    <p class="wi-sum"><span class="${c >= 0 ? "up" : "down"}">${esc(t("whatif.change", { sign: c >= 0 ? "+" : "−", n: num1(Math.abs(c)) }))}</span> · ${esc(tierLine)}</p>
    ${note}${warns}`;
}
