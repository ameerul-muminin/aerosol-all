// F4 All wards: early warnings, tier counts, ranking with +2/+3/+4 heat strip (ward × horizon heatmap),
// by-city bars, CSV. Data: GET /api/ranking and /api/warnings (cached by api.js).
import { api } from "../api.js";
import { t } from "../i18n.js";
import { num0, num1, dateLong, cityName, wardName, tierHTML, esc } from "../format.js";
import { showError, ticket, segment } from "../ui.js";

const TIERS = ["Critical", "High", "Medium", "Low"];
// heat steps (expected cases) -> --heat-1..7; text turns light from step 5
const HEAT_STEPS = [1, 3, 5, 8, 11, 15];
const heat = (v) => { const i = HEAT_STEPS.findIndex((x) => v < x); return i === -1 ? 7 : i + 1; };

export async function render(root, { state: s, meta, set, retry }) {
  const current = ticket(root);
  if (!root.querySelector(".rank")) {
    root.innerHTML = `<div class="head"><h1 class="display">${esc(t("wards.title"))}</h1><p><span class="skel">${esc(t("state.loading"))}</span></p></div>
      <div class="grid" aria-busy="true"><section class="card span-12"><span class="skel" style="height:320px;width:100%"></span></section></div>`;
  }
  let r, warn;
  try {
    [r, warn] = await Promise.all([api.ranking(s.as_of, s.h, s.city), api.warnings(s.as_of, s.city)]);
  } catch (err) {
    if (current()) showError(root, err, retry);
    return;
  }
  if (!current()) return;

  const total = r.wards.length;
  const hz = meta.horizons;
  root.innerHTML = `
  <div class="head">
    <h1 class="display">${esc(t("wards.title"))}</h1>
    <p>${esc(t("wards.subtitle", { week: dateLong(r.forecast_week) }))}</p>
  </div>
  <div class="toolbar">
    <div class="field"><span id="wh-l">${esc(t("outlook.horizon_label"))}</span>
      <div class="seg" role="group" aria-labelledby="wh-l"><span class="thumb"></span>
        ${hz.map((h) => `<button type="button" data-v="${h}">${esc(t("outlook.plus_weeks", { h: num0(h) }))}</button>`).join("")}</div></div>
    <label class="field"><span>${esc(t("ctl.city"))}</span><select id="w-city">
      <option value="">${esc(t("ctl.city_all"))}</option>
      ${meta.cities.map((c) => `<option value="${esc(c.city)}"${c.city === s.city ? " selected" : ""}>${esc(cityName(c.city))}</option>`).join("")}
    </select></label>
    <button type="button" class="btn small" id="w-csv">${esc(t("wards.csv"))}</button>
  </div>
  <div class="grid">
    <section class="card span-12">${warnings(warn, s)}</section>
    ${TIERS.map((k) => `<section class="card tile span-3">${tierHTML(k)}<b>${num0(r.tier_counts[k])}</b>
      <span class="of">${esc(t("wards.count_wards", { n: num0(total) }))}</span></section>`).join("")}
    <section class="card span-8">${total ? table(r, s) : `<p>${esc(t("wards.empty"))}</p>`}</section>
    <section class="card span-4"><h2>${esc(t("wards.by_city"))}</h2>${cities(r)}</section>
  </div>`;

  segment(root.querySelector(".seg"), s.h, (v) => set({ h: Number(v) }));
  root.querySelector("#w-city").onchange = (e) => set({ city: e.target.value });
  root.querySelector("#w-csv").onclick = () => downloadCsv(r);
  const more = root.querySelector("#warn-more");
  if (more) more.onclick = () => { root.querySelectorAll(".warn li[hidden]").forEach((li) => (li.hidden = false)); more.remove(); };
  root.querySelectorAll(".rank a, .warn a").forEach((a) => (a.onclick = (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;   // let "open in new tab" work
    e.preventDefault();
    set({ ward: a.dataset.ward, view: "outlook", ...(a.closest(".warn") ? { h: Number(new URLSearchParams(a.search).get("h")) } : {}) });
  }));
}

/** Wards getting worse over the next 4 weeks, worst first: tier and cases at +2 → +3 → +4, plus the action. */
const WARN_SHOWN = 8;
function warnings(w, s) {
  const head = `<div class="warn-head"><h2>${esc(t("warn.title"))}</h2>
    <span>${esc(t("warn.count", { n: num0(w.wards.length), total: num0(w.n_wards) }))}</span></div>`;
  if (!w.wards.length) return `${head}<p class="warn-none">${esc(t("warn.none"))}</p>`;
  const items = w.wards.map((x, i) => {
    const name = wardName(x.ward_id);
    const href = `?${new URLSearchParams({ ward: x.ward_id, as_of: s.as_of, h: x.peak_horizon })}#outlook`;
    const steps = x.steps.map((st) => `<span class="st"><em>${esc(t("outlook.plus_weeks", { h: num0(st.horizon) }))}</em>
      ${tierHTML(st.risk_tier)}<b>${esc(t("warn.cases", { n: num1(st.predicted_cases) }))}</b></span>`).join(`<span class="arr" aria-hidden="true">→</span>`);
    return `<li${i >= WARN_SHOWN ? " hidden" : ""}>
      <div class="who"><a href="${href}" data-ward="${x.ward_id}" aria-label="${esc(t("wards.open", { ward: `${name}, ${cityName(x.city)}` }))}">${esc(name)}</a>
        <span class="city">${esc(cityName(x.city))} · ${esc(t(`warn.${x.reason}`))}</span></div>
      <div class="steps">${steps}</div>
      <p class="act">${esc(t(`action.${x.peak_tier}`))}</p></li>`;
  }).join("");
  const more = w.wards.length > WARN_SHOWN
    ? `<button type="button" class="btn small" id="warn-more">${esc(t("warn.show_all", { n: num0(w.wards.length) }))}</button>` : "";
  return `${head}<ol class="warn">${items}</ol>${more}<p class="acc-note">${esc(t("warn.rule"))}</p>`;
}

function table(r, s) {
  const rows = r.wards.map((w, i) => {
    const bh = w.by_horizon;
    const href = `?${new URLSearchParams({ ward: w.ward_id, as_of: s.as_of, h: s.h })}#outlook`;
    const name = wardName(w.ward_id);
    const cells = Object.keys(bh).map((k) => { const lvl = heat(bh[k]); return `<span class="h${lvl}${lvl >= 5 ? " dk" : ""}">${num1(bh[k])}</span>`; }).join("");
    const aria = t("wards.heat_aria", { ward: name, a: num1(bh["2"]), b: num1(bh["3"]), c: num1(bh["4"]) });
    return `<tr>
      <td class="r">${num0(i + 1)}</td>
      <td><a href="${href}" data-ward="${w.ward_id}" aria-label="${esc(t("wards.open", { ward: `${name}, ${cityName(w.city)}` }))}">${esc(name)}</a><span class="city">${esc(cityName(w.city))}</span></td>
      <td class="c">${num1(w.predicted_cases)}</td>
      <td>${tierHTML(w.risk_tier)}</td>
      <td class="num">${num0(w.range_low)}–${num0(w.range_high)}</td>
      <td><span class="heat" role="img" aria-label="${esc(aria)}">${cells}</span></td>
    </tr>`;
  }).join("");
  return `<div class="tbl-wrap"><table class="rank">
    <thead><tr><th scope="col">${esc(t("wards.rank"))}</th><th scope="col">${esc(t("wards.ward"))}</th><th scope="col" class="c">${esc(t("wards.cases"))}</th>
      <th scope="col">${esc(t("wards.tier"))}</th><th scope="col">${esc(t("wards.range"))}</th><th scope="col">${esc(t("wards.heat"))}</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function cities(r) {
  return r.by_city.map((c) => {
    const n = TIERS.reduce((a, k) => a + c[k], 0) || 1;
    const segs = TIERS.filter((k) => c[k]).map((k, i) => `<i class="bg-${k}" style="width:${(c[k] / n) * 100}%;animation-delay:${i * 30}ms" title="${esc(t(`tier.${k}`))}: ${num0(c[k])}"></i>`).join("");
    const label = TIERS.filter((k) => c[k]).map((k) => `${t(`tier.${k}`)} ${num0(c[k])}`).join(", ");
    return `<div class="cityrow"><div class="top"><b>${esc(cityName(c.city))}</b><span>${esc(t("wards.count_wards", { n: num0(n) }))}</span></div>
      <div class="stack" role="img" aria-label="${esc(`${cityName(c.city)}: ${label}`)}">${segs}</div></div>`;
  }).join("") + `<div class="legend" style="margin:14px 0 0">${TIERS.map((k) => tierHTML(k)).join("")}</div>`;
}

function downloadCsv(r) {
  const head = ["rank", "ward_id", "city", "forecast_week", "horizon_weeks", "predicted_cases", "range_low", "range_high", "risk_tier", "cases_plus2", "cases_plus3", "cases_plus4"];
  const lines = r.wards.map((w, i) => [i + 1, w.ward_id, w.city, w.forecast_week, r.horizon, w.predicted_cases, w.range_low, w.range_high, w.risk_tier,
    w.by_horizon["2"], w.by_horizon["3"], w.by_horizon["4"]].join(","));
  const blob = new Blob([`${head.join(",")}\n${lines.join("\n")}\n`], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `dengue_ranking_${r.forecast_week}_h${r.horizon}.csv` });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
