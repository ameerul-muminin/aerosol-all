// F5 Model accuracy (validation dashboard): test-year MAE/RMSE per horizon vs the seasonal-average baseline,
// error by city, 80% range coverage, and one ward's forecast vs what happened with the 80% band.
// Data: GET /api/metrics (from reports/metrics.json + test predictions) and GET /api/history (cached by api.js).
import { api } from "../api.js";
import { t, getLang } from "../i18n.js";
import { num0, num1, num2, dateLong, cityName, wardFull, esc } from "../format.js";
import { showError, ticket, segment } from "../ui.js";

const pct = (x) => `${num0(x * 100)}%`;
/** A year without digit grouping ("2024", "২০২৪"), from "2024" or "2024-01-01". */
const year = (v) => new Intl.NumberFormat(getLang() === "bn" ? "bn-BD" : "en-GB", { useGrouping: false }).format(Number(String(v).slice(0, 4)));
let offResize = null;   // the history chart re-draws to its card width

export async function render(root, { state: s, meta, set, retry }) {
  const current = ticket(root);
  if (!root.querySelector(".acc")) {
    root.innerHTML = `<div class="head"><h1 class="display">${esc(t("acc.title"))}</h1><p><span class="skel">${esc(t("state.loading"))}</span></p></div>
      <div class="grid" aria-busy="true"><section class="card span-12"><span class="skel" style="height:320px;width:100%"></span></section></div>`;
  }
  let m, hist, city, lag;
  try {
    city = meta.cities.find((c) => c.wards.includes(s.ward)).city;
    [m, hist, lag] = await Promise.all([api.metrics(), api.history(s.ward, s.h), api.lagProfile(s.city)]);
  } catch (err) {
    if (current()) showError(root, err, retry);
    return;
  }
  if (!current()) return;

  const hz = m.horizons.find((x) => x.horizon === s.h) || m.horizons[0];
  const years = (m.split.test.match(/\d{4}/g) || []);
  const period = years.length ? `${year(years[0])}–${year(years[years.length - 1])}` : m.split.test;
  const pts = hist.points;
  const inside = pts.filter((p) => p.actual >= p.low && p.actual <= p.high).length;

  root.innerHTML = `
  <div class="head acc">
    <h1 class="display">${esc(t("acc.title"))}</h1>
    <p>${esc(t("acc.subtitle", { period }))}</p>
  </div>
  <div class="toolbar">
    <div class="field"><span id="ah-l">${esc(t("outlook.horizon_label"))}</span>
      <div class="seg" role="group" aria-labelledby="ah-l"><span class="thumb"></span>
        ${meta.horizons.map((h) => `<button type="button" data-v="${h}">${esc(t("outlook.plus_weeks", { h: num0(h) }))}</button>`).join("")}</div></div>
  </div>
  <div class="grid">
    ${tile(t("acc.mae"), num2(hz.mae), t("acc.vs_base", { n: num2(hz.baseline_mae) }), better(hz.mae, hz.baseline_mae))}
    ${tile(t("acc.rmse"), num2(hz.rmse), t("acc.vs_base", { n: num2(hz.baseline_rmse) }), better(hz.rmse, hz.baseline_rmse))}
    ${tile(t("acc.coverage"), pct(hz.interval80_coverage), t("acc.coverage_note"))}
    ${hz.critical_flagged_high_or_above == null ? "" : tile(t("acc.critical"), pct(hz.critical_flagged_high_or_above), t("acc.critical_note"))}
    <section class="card span-7"><h2>${esc(t("acc.by_h_title"))}</h2>${byHorizon(m.horizons, s.h)}</section>
    <section class="card span-5"><h2>${esc(t("acc.by_city_title", { h: num0(hz.horizon) }))}</h2>${byCity(hz.by_city)}</section>
    <section class="card span-12">
      <h2>${esc(t("acc.hist_title", { ward: wardFull(s.ward, city), h: num0(s.h) }))}</h2>
      <div class="legend">
        <span><i style="background:var(--ink)"></i>${esc(t("acc.lg_pred"))}</span>
        <span><i style="background:var(--raises);border-radius:50%"></i>${esc(t("acc.lg_actual"))}</span>
        <span><i style="background:var(--accent-soft);box-shadow:inset 0 0 0 1px var(--accent)"></i>${esc(t("acc.lg_band"))}</span>
        <span><i style="box-shadow:inset 0 0 0 1.5px var(--raises);border-radius:50%"></i>${esc(t("acc.lg_out"))}</span>
      </div>
      <div class="hist" role="img" aria-label="${esc(t("acc.hist_aria", { n: num0(pts.length), inside: num0(inside) }))}"></div>
      <p class="hist-sum">${esc(t("acc.hist_sum", { inside: num0(inside), n: num0(pts.length), p: pct(pts.length ? inside / pts.length : 0) }))}</p>
    </section>
    <section class="card span-12">${clock(lag, meta, s)}</section>
  </div>`;

  segment(root.querySelector(".seg"), s.h, (v) => set({ h: Number(v) }));
  root.querySelector("#clk-city").onchange = (e) => set({ city: e.target.value });
  const box = root.querySelector(".hist");
  drawHistory(box, pts);
  if (offResize) offResize();
  let raf = 0;
  const onResize = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { if (box.isConnected) drawHistory(box, pts); else offResize(); }); };
  window.addEventListener("resize", onResize);
  offResize = () => { window.removeEventListener("resize", onResize); offResize = null; };
}

/** Deployed model in plain words ("Weighted ensemble of 3 models"); falls back to the API name. */
const modelName = (h) => t(`acc.model_${h.model_type}`, { n: num0((h.model_members || []).length) }, h.model_type || "");

function better(model, base) {
  if (!(base > 0)) return "";
  const d = (base - model) / base;
  return d > 0 ? t("acc.lower_by", { p: pct(d) }) : t("acc.not_lower");
}

function tile(label, value, note, extra = "") {
  return `<section class="card tile span-3"><span class="k">${esc(label)}</span><b>${esc(value)}</b>
    <span class="of">${esc(note)}</span>${extra ? `<span class="gain">${esc(extra)}</span>` : ""}</section>`;
}

/** Paired bars per horizon: model vs seasonal average, for MAE and RMSE. One shared scale. */
function byHorizon(hs, sel) {
  const max = Math.max(...hs.flatMap((h) => [h.mae, h.rmse, h.baseline_mae, h.baseline_rmse]));
  const bar = (v, cls, i) => `<i class="${cls}" style="width:${(v / max) * 100}%;animation-delay:${i * 30}ms"></i><span>${num2(v)}</span>`;
  const rows = hs.map((h, i) => `
    <div class="pair${h.horizon === sel ? " on" : ""}" role="img"
      aria-label="${esc(t("acc.pair_aria", { h: num0(h.horizon), mae: num2(h.mae), bmae: num2(h.baseline_mae), rmse: num2(h.rmse), brmse: num2(h.baseline_rmse) }))}">
      <span class="hl">${esc(t("outlook.plus_weeks", { h: num0(h.horizon) }))}<small>${esc(modelName(h))}</small></span>
      <div class="m"><em>${esc(t("acc.mae_short"))}</em><div class="b">${bar(h.mae, "mod", i)}</div><div class="b">${bar(h.baseline_mae, "base", i)}</div></div>
      <div class="m"><em>${esc(t("acc.rmse_short"))}</em><div class="b">${bar(h.rmse, "mod", i)}</div><div class="b">${bar(h.baseline_rmse, "base", i)}</div></div>
    </div>`).join("");
  return `<div class="legend">
      <span><i style="background:var(--ink)"></i>${esc(t("acc.lg_model"))}</span>
      <span><i style="background:var(--bar)"></i>${esc(t("acc.lg_base"))}</span></div>
    <div class="pairs">${rows}</div><p class="acc-note">${esc(t("acc.lower_better"))}</p>`;
}

function byCity(rows) {
  const max = Math.max(...rows.map((r) => r.mae));
  return `<table class="rank cerr"><thead><tr><th scope="col">${esc(t("ctl.city"))}</th><th aria-hidden="true"></th>
      <th scope="col" class="c">${esc(t("acc.mae_short"))}</th><th scope="col" class="c">${esc(t("acc.rmse_short"))}</th></tr></thead><tbody>
    ${[...rows].sort((a, b) => a.mae - b.mae).map((r, i) => `<tr><td>${esc(cityName(r.city))}</td>
      <td class="bt"><i style="width:${(r.mae / max) * 100}%;animation-delay:${i * 30}ms"></i></td>
      <td class="c">${num2(r.mae)}</td><td class="c">${num2(r.rmse)}</td></tr>`).join("")}
  </tbody></table><p class="acc-note">${esc(t("acc.city_note"))}</p>`;
}

/** Breeding clock: how strongly rain k weeks ago tracks cases now (from /api/lag_profile).
 *  The 2–4 week window is the brief's breeding lag; the bars are whatever the data says. */
const WINDOW = [2, 4];
function clock(lag, meta, s) {
  const max = Math.max(...lag.lags.map((x) => x.corr), 0.01);
  const bars = lag.lags.map((x, i) => {
    const inWin = x.weeks >= WINDOW[0] && x.weeks <= WINDOW[1];
    const pk = x.weeks === lag.peak_weeks;
    const hgt = (Math.max(0, x.corr) / max) * 70;   // labels take ~26% of the column height
    return `<div class="c${inWin ? " bw" : ""}${pk ? " pk" : ""}"><span class="mm">${num2(Math.abs(x.corr) < 0.005 ? 0 : x.corr)}</span>
      <i style="height:${hgt}%;animation-delay:${i * 20}ms"></i><em>${num0(x.weeks)}</em></div>`;
  }).join("");
  const aria = t("acc.clock_aria", { list: lag.lags.map((x) => `${num0(x.weeks)}: ${num2(x.corr)}`).join(", ") });
  return `<div class="clk-head"><h2>${esc(t("acc.clock_title"))}</h2>
      <label class="field"><span>${esc(t("ctl.city"))}</span><select id="clk-city">
        <option value="">${esc(t("ctl.city_all"))}</option>
        ${meta.cities.map((c) => `<option value="${esc(c.city)}"${c.city === s.city ? " selected" : ""}>${esc(cityName(c.city))}</option>`).join("")}
      </select></label></div>
    <p class="clk-sum">${esc(t("acc.clock_sum", { w: num0(lag.peak_weeks), c: num2(lag.peak_corr) }))}</p>
    <div class="legend">
      <span><i style="background:var(--accent)"></i>${esc(t("acc.clock_window"))}</span>
      <span><i style="background:var(--bar)"></i>${esc(t("acc.clock_other"))}</span>
    </div>
    <div class="rain clk" style="--n:${lag.lags.length}" role="img" aria-label="${esc(aria)}">${bars}</div>
    <p class="clk-axis">${esc(t("acc.clock_axis"))}</p>
    <p class="acc-note">${esc(t("acc.clock_note", { from: year(lag.period.from), to: year(lag.period.to) }))}</p>`;
}

/** SVG line chart sized to its box: 80% band, predicted line, actual dots, hover readout. */
function drawHistory(box, pts) {
  if (!pts.length) { box.innerHTML = ""; return; }
  const W = Math.max(300, box.clientWidth);
  const H = W < 600 ? 200 : 260;
  const L = 34, R = 8, T = 8, B = 26;
  const top = Math.max(4, ...pts.map((p) => Math.max(p.high, p.actual)));
  const step = top > 20 ? 10 : top > 8 ? 5 : 2;
  const yMax = Math.ceil(top / step) * step;
  const x = (i) => L + (i / Math.max(1, pts.length - 1)) * (W - L - R);
  const y = (v) => T + (1 - v / yMax) * (H - T - B);
  const line = (k) => pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`).join("");
  const band = `${pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.high).toFixed(1)}`).join("")}`
    + `${[...pts].reverse().map((p, j) => `L${x(pts.length - 1 - j).toFixed(1)},${y(p.low).toFixed(1)}`).join("")}Z`;
  const grid = [];
  for (let v = 0; v <= yMax; v += step) {
    grid.push(`<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="gl"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${num0(v)}</text>`);
  }
  // x labels: the first week of each quarter (Jan, Apr, Jul, Oct), fewer on narrow screens
  const every = W < 600 ? 6 : 3;
  const ticks = pts.map((p, i) => [p, i]).filter(([p], i) => {
    const d = new Date(`${p.week}T00:00:00`);
    const prev = i ? new Date(`${pts[i - 1].week}T00:00:00`) : null;
    return d.getMonth() % every === 0 && (!prev || prev.getMonth() !== d.getMonth());
  }).map(([p, i]) => `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${esc(dateShortYear(p.week))}</text>`);
  const r = W < 600 ? 2 : 2.6;
  box.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true">
    ${grid.join("")}
    <path d="${band}" class="band"/>
    <path d="${line("predicted")}" class="pl"/>
    ${pts.map((p, i) => `<circle cx="${x(i).toFixed(1)}" cy="${y(p.actual).toFixed(1)}" r="${r}" class="${p.actual < p.low || p.actual > p.high ? "out" : ""}"/>`).join("")}
    ${ticks.join("")}
    <line class="cur" y1="${T}" y2="${H - B}" x1="0" x2="0" visibility="hidden"/>
  </svg><div class="tip" hidden></div>`;

  const svg = box.querySelector("svg");
  const cur = box.querySelector(".cur");
  const tip = box.querySelector(".tip");
  svg.onpointermove = (e) => {
    const rect = svg.getBoundingClientRect();
    const i = Math.round(((e.clientX - rect.left - L) / (W - L - R)) * (pts.length - 1));
    if (i < 0 || i >= pts.length) { svg.onpointerleave(); return; }
    const p = pts[i];
    cur.setAttribute("x1", x(i)); cur.setAttribute("x2", x(i)); cur.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.innerHTML = `<b>${esc(dateLong(p.week))}</b><span>${esc(t("acc.tip_actual", { n: num0(p.actual) }))}</span>
      <span>${esc(t("acc.tip_pred", { n: num1(p.predicted) }))}</span><span>${esc(t("acc.tip_range", { low: num1(p.low), high: num1(p.high) }))}</span>`;
    const left = Math.min(Math.max(x(i) + 12, 0), W - tip.offsetWidth - 4);
    tip.style.transform = `translate(${x(i) + 12 + tip.offsetWidth > W ? x(i) - tip.offsetWidth - 12 : left}px, 8px)`;
  };
  svg.onpointerleave = () => { tip.hidden = true; cur.setAttribute("visibility", "hidden"); };
}

/** Axis label: "Apr", and "Jan 2025" at the turn of each year (in the active language). */
function dateShortYear(iso) {
  const d = new Date(`${iso}T00:00:00`);
  const opts = d.getMonth() === 0 ? { month: "short", year: "numeric" } : { month: "short" };
  return d.toLocaleDateString(getLang() === "bn" ? "bn-BD" : "en-GB", opts);
}
