// Landing "Check a ward": ward + week + weeks ahead, optional weather for that week -> forecast.
// Empty weather -> GET /api/forecast (recorded weather). Any weather typed -> POST /api/what_if with
// weeks: 1 (only that week is replaced), shown next to the recorded-weather forecast.
// The dataset ends at meta.as_of.max, so the default week is "the latest week in the data", said plainly.
import { t } from "./i18n.js";
import { api } from "./api.js";
import { num0, num1, signed1, dateLong, cityName, wardName, wardFull, tierHTML, driverName, esc } from "./format.js";

const $ = (id) => document.getElementById(id);
const FIELDS = [["rainfall_mm", "rain", "rain_unit"], ["avg_temp_c", "temp", "temp_unit"], ["relative_humidity_pct", "hum", "hum_unit"]];
let meta = null;
let last = null;      // { kind: "forecast" | "what_if", data, input } -> re-rendered on language change
let busy = false;
/** Range ends: whole cases, or one decimal when the whole range is under 1 case (so it never reads "0–0"). */
const rng = (v, high) => (high < 1 ? num1(v) : num0(v));

/** Build (or rebuild after a language change) the form, keeping what the person already chose or typed. */
export function renderCheck(m) {
  meta = m;
  const form = $("check-form");
  const keep = form.elements.length ? readForm() : null;
  const weeks = [];
  for (let d = new Date(`${meta.as_of.max}T00:00:00Z`); d >= new Date(`${meta.as_of.min}T00:00:00Z`); d.setUTCDate(d.getUTCDate() - 7)) {
    weeks.push(d.toISOString().slice(0, 10));
  }
  form.innerHTML = `
    <label class="field"><span>${esc(t("ctl.ward"))}</span><select id="cw">
      ${meta.cities.map((c) => `<optgroup label="${esc(cityName(c.city))}">${c.wards.map((w) => `<option value="${w}">${esc(wardName(w))}</option>`).join("")}</optgroup>`).join("")}
    </select></label>
    <label class="field"><span>${esc(t("landing.check_week"))}</span><select id="cweek">
      ${weeks.map((w, i) => `<option value="${w}">${esc(i === 0 ? t("landing.check_latest", { date: dateLong(w) }) : dateLong(w))}</option>`).join("")}
    </select></label>
    <label class="field"><span>${esc(t("outlook.horizon_label"))}</span><select id="ch">
      ${meta.horizons.map((h) => `<option value="${h}">${esc(t("outlook.plus_weeks", { h: num0(h) }))}</option>`).join("")}
    </select></label>
    <fieldset class="wx">
      <legend>${esc(t("landing.check_wx"))}</legend>
      ${FIELDS.map(([k, lab, unit]) => `<label class="field"><span>${esc(t(`whatif.${lab}`))} <span class="unit">${esc(t(`whatif.${unit}`))}</span></span>
        <input type="number" id="c-${k}" inputmode="decimal" step="any" placeholder="${esc(t("landing.check_recorded"))}"></label>`).join("")}
    </fieldset>
    <div class="row"><button type="submit" class="btn primary" id="c-go">${esc(t("landing.check_go"))}</button>
      <span class="hint">${esc(t("landing.check_hint"))}</span></div>`;
  const ward = keep ? keep.ward : meta.cities[0].wards[0];
  $("cw").value = ward;
  $("cweek").value = keep ? keep.as_of : meta.as_of.max;
  $("ch").value = String(keep ? keep.horizon : 3);
  if (keep) FIELDS.forEach(([k]) => { if (keep.wx[k] !== undefined) $(`c-${k}`).value = keep.wx[k]; });
  form.onsubmit = (e) => { e.preventDefault(); run(); };
  paint();
}

function readForm() {
  const wx = {};
  FIELDS.forEach(([k]) => {
    const v = $(`c-${k}`).value.trim();
    if (v !== "") wx[k] = Number(v);
  });
  return { ward: $("cw").value, as_of: $("cweek").value, horizon: Number($("ch").value), wx };
}

async function run() {
  if (busy) return;
  const input = readForm();
  busy = true;
  $("c-go").disabled = true;
  $("c-go").textContent = t("whatif.running");
  $("check-out").setAttribute("aria-busy", "true");
  try {
    if (Object.keys(input.wx).length) {
      const data = await api.whatIf({ ward_id: input.ward, as_of: input.as_of, horizon: input.horizon, weeks: 1, ...input.wx });
      last = { kind: "what_if", data, input };
    } else {
      const data = await api.forecast(input.ward, input.as_of, input.horizon);
      last = { kind: "forecast", data, input };
    }
  } catch (err) {
    last = { kind: "error", message: err.message, input };
  } finally {
    busy = false;
    $("c-go").disabled = false;
    $("c-go").textContent = t("landing.check_go");
    $("check-out").removeAttribute("aria-busy");
  }
  paint();
}

/** The result card (or its empty/error state), in the current language. */
function paint() {
  const out = $("check-out");
  if (!last) { out.innerHTML = `<p class="empty">${esc(t("landing.check_empty"))}</p>`; return; }
  if (last.kind === "error") { out.innerHTML = `<p class="inline-error" role="alert">${esc(last.message)}</p>`; return; }
  const { data, input } = last;
  const wi = last.kind === "what_if";
  const f = wi ? data.scenario : data;
  const city = data.city;
  const href = `/app/?${new URLSearchParams({ ward: input.ward, as_of: input.as_of, h: input.horizon })}#outlook`;
  const drivers = (f.top_drivers || []).slice(0, 3).map((d) => {
    return `<li><span>${esc(driverName(d.label))}</span><b class="${d.effect >= 0 ? "up" : "down"}">${esc(t("landing.check_effect", { n: signed1(d.effect) }))}</b></li>`;
  }).join("");
  out.innerHTML = `
    <p class="where">${esc(wardFull(input.ward, city))} · ${esc(t("landing.check_for", { week: dateLong(data.forecast_week), h: num0(input.horizon) }))}</p>
    ${wi ? `<p class="tag">${esc(t("landing.check_with_yours"))}</p>` : ""}
    <div class="res"><b class="n">${num1(f.predicted_cases)}</b><div>
      <span class="lbl">${esc(t("outlook.expected"))}</span>
      <div class="tierline">${tierHTML(f.risk_tier, { risk: true })}</div>
      <p class="rng">${esc(t("outlook.range", { low: rng(f.range_low, f.range_high), high: rng(f.range_high, f.range_high) }))}</p></div></div>
    ${wi ? `<p class="cmp">${esc(t("landing.check_recorded_was", { n: num1(data.baseline.predicted_cases) }))} ${tierHTML(data.baseline.risk_tier)}</p>` : ""}
    ${wi && data.warnings.length ? `<ul class="wi-warn">${data.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
    ${drivers ? `<h3>${esc(t("landing.check_why"))}</h3><ul class="why">${drivers}</ul>` : ""}
    <a class="btn" href="${href}">${esc(t("landing.check_open"))}</a>`;
}
