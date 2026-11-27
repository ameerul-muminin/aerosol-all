// R5 Supply allocation: live scarcity controls (stock slider, crews, demand basis, city), deficit alert,
// totals, ward table in priority order, order plan for weeks +2..+4 (bonus B1).
// Data: POST /api/allocate, GET /api/supply_plan. Every number comes from these answers.
import { api } from "../api.js";
import { t } from "../i18n.js";
import { num0, num1, dateLong, cityName, wardName, tierHTML, esc } from "../format.js";
import { showError, ticket, segment, debounce, loadingHTML, rangeField, alertIcon, critBadge, inlineError, downloadCsv } from "../ui.js";

const ITEMS = ["saline", "kits", "beds"];
const SHOWN = 20;
let showAll = false;   // "Show all" survives slider changes

export async function render(root, { state, meta, save, retry }) {
  const s = { ...state };
  const cur = ticket(root);
  root.innerHTML = loadingHTML(t("alloc.title"));
  let r, plan;
  try {
    [r, plan] = await Promise.all([api.allocate(body(s)), api.supplyPlan(s.as_of, s.city)]);
  } catch (err) {
    if (cur()) showError(root, err, retry);
    return;
  }
  if (!cur()) return;

  const a = r.assumptions;
  const [smin, smax] = a.stock.stock_pct_range.value;
  const pct = (v) => `${num0(v)}%`;
  root.innerHTML = `
  <div class="head"><h1 class="display">${esc(t("alloc.title"))}</h1><p id="a-sub"></p></div>
  <div class="toolbar">
    <div class="field"><span id="ah-l">${esc(t("outlook.horizon_label"))}</span>
      <div class="seg" id="a-h" role="group" aria-labelledby="ah-l"><span class="thumb"></span>
        ${meta.horizons.map((h) => `<button type="button" data-v="${h}">${esc(t("outlook.plus_weeks", { h: num0(h) }))}</button>`).join("")}</div></div>
    ${rangeField("a-stock", t("alloc.stock"), { min: smin, max: smax, step: 5, value: s.stock, fmt: pct })}
    <label class="field"><span>${esc(t("alloc.crews"))}</span>
      <input type="number" id="a-crews" min="0" step="1" inputmode="numeric" value="${s.crews !== "" ? esc(s.crews) : r.crews}"></label>
    <div class="field"><span id="ab-l">${esc(t("alloc.demand"))}</span>
      <div class="seg" id="a-basis" role="group" aria-labelledby="ab-l"><span class="thumb"></span>
        ${a.demand.basis.allowed.map((b) => `<button type="button" data-v="${b}">${esc(t(`alloc.basis_${b}`))}</button>`).join("")}</div></div>
    <label class="field"><span>${esc(t("ctl.city"))}</span><select id="a-city">
      <option value="">${esc(t("ctl.city_all"))}</option>
      ${meta.cities.map((c) => `<option value="${esc(c.city)}"${c.city === s.city ? " selected" : ""}>${esc(cityName(c.city))}</option>`).join("")}
    </select><span class="cap">${esc(t("alloc.totals_cover", { n: num0(r.totals.wards) }))}</span></label>
  </div>
  <p class="inline-error" role="alert" id="a-err" hidden></p>
  <div class="out" id="a-out"></div>`;

  const out = root.querySelector("#a-out");
  const err = root.querySelector("#a-err");
  paint(out, r, plan, s);

  async function load() {
    const mine = ticket(out);   // only the latest answer paints
    out.setAttribute("aria-busy", "true");
    try {
      const [r2, p2] = await Promise.all([api.allocate(body(s)), api.supplyPlan(s.as_of, s.city)]);
      if (!mine() || !out.isConnected) return;
      inlineError(err, null);
      paint(out, r2, p2, s);
    } catch (e) {
      if (mine()) inlineError(err, e);
    } finally {
      if (mine()) out.removeAttribute("aria-busy");
    }
  }
  const change = (patch) => { Object.assign(s, patch); save(patch); load(); };
  const later = debounce(change, 200);

  segment(root.querySelector("#a-h"), s.h, (v) => change({ h: Number(v) }));
  segment(root.querySelector("#a-basis"), s.basis || "predicted_cases", (v) => change({ basis: v === "predicted_cases" ? "" : v }));
  const stock = root.querySelector("#a-stock");
  stock.oninput = () => {
    root.querySelector("#a-stock-out").textContent = pct(stock.value);
    later({ stock: Number(stock.value) });
  };
  root.querySelector("#a-crews").oninput = (e) => {
    const v = e.target.value.trim();
    if (v !== "" && !/^\d+$/.test(v)) return;   // the browser shows the field as invalid; wait for a whole number
    later({ crews: v });
  };
  root.querySelector("#a-city").onchange = (e) => change({ city: e.target.value });
}

function body(s) {
  const b = { as_of: s.as_of, horizon: s.h, stock_pct: s.stock };
  if (s.crews !== "") b.crews = Number(s.crews);
  if (s.basis) b.demand_basis = s.basis;
  if (s.city) b.city = s.city;
  return b;
}

function paint(out, r, plan, s) {
  const tot = r.totals;
  const items = tot.items;
  out.closest(".main").querySelector("#a-sub").textContent =
    t("alloc.subtitle", { week: dateLong(r.forecast_week), h: num0(r.horizon) });

  const banner = tot.critical_wards > 0
    ? `<div class="alert" role="alert">${alertIcon}<span>${esc(t("alloc.alert", {
      n: num0(tot.critical_wards), total: num0(tot.wards),
      saline: num0(items.saline.deficit), kits: num0(items.kits.deficit), beds: num0(items.beds.deficit) }))}</span></div>`
    : `<p class="calm">${esc(t("alloc.all_supplied"))}</p>`;

  const tiles = ITEMS.map((k) => {
    const it = items[k];
    return `<section class="card tile span-3"><span class="k">${esc(t(`alloc.item_${k}`))}</span><b>${num0(it.allocated)}</b>
      <span class="of">${esc(t("alloc.of", { n: num0(it.required) }))}</span>
      ${it.deficit > 0 ? `<span class="sub deficit">${esc(t("alloc.short_in", { n: num0(it.deficit), w: num0(it.wards_short) }))}</span>` : ""}</section>`;
  }).join("") + `<section class="card tile span-3"><span class="k">${esc(t("alloc.item_crews"))}</span><b>${num0(tot.crews.assigned)}</b>
      <span class="of">${esc(t("alloc.of", { n: num0(tot.crews.available) }))}</span>
      <span class="sub of">${esc(t("alloc.unused", { n: num0(tot.crews.unused) }))}</span></section>`;

  out.innerHTML = `${banner}
  <div class="grid">
    ${tiles}
    <section class="card span-12">
      <div class="warn-head"><h2>${esc(t("alloc.table_title"))}</h2><span>${esc(t("alloc.table_note"))}</span></div>
      ${table(r, s)}
      <div class="more-row">
        ${r.wards.length > SHOWN ? `<button type="button" class="btn small" id="a-all">${esc(showAll ? t("alloc.show_less") : t("alloc.show_all", { n: num0(r.wards.length) }))}</button>` : ""}
        <button type="button" class="btn small" id="a-csv">${esc(t("wards.csv"))}</button>
      </div>
    </section>
    <section class="card span-7">${orderPlan(plan, s)}</section>
    <section class="card span-5"><h2>${esc(t("alloc.how_title"))}</h2>${how(r.assumptions)}</section>
  </div>`;

  const all = out.querySelector("#a-all");
  if (all) all.onclick = () => { showAll = !showAll; paint(out, r, plan, s); out.querySelector("#a-all").focus(); };
  out.querySelector("#a-csv").onclick = () => csv(r);
}

function table(r, s) {
  const cautious = s.basis === "range_high";
  const rows = (showAll ? r.wards : r.wards.slice(0, SHOWN)).map((w) => {
    const items = ITEMS.map((k) => {
      const d = w[`${k}_deficit`];
      return `<td class="num${d > 0 ? " deficit" : ""}">${num0(w[`${k}_allocated`])} / ${num0(w[`${k}_required`])}${d > 0 ? `<span class="sr-only"> ${esc(t("alloc.short", { n: num0(d) }))}</span>` : ""}</td>`;
    }).join("");
    const f = w.fulfilled_pct;
    return `<tr>
      <td class="r">${num0(w.priority_rank)}</td>
      <td><b>${esc(wardName(w.ward_id))}</b><span class="city">${esc(cityName(w.city))}</span></td>
      <td>${tierHTML(w.risk_tier)}</td>
      <td class="num">${num0(w.population_density_sqkm)}</td>
      <td class="num">${num1(w.predicted_cases)}${cautious ? `<span class="sub-line">${esc(t("alloc.plan_for", { n: num0(w.demand_cases) }))}</span>` : ""}</td>
      ${items}
      <td><span class="mini"><span class="hbar"><i class="${f < 100 ? "short" : ""}" style="transform:scaleX(${Math.min(f, 100) / 100})"></i></span>${num0(f)}%</span></td>
      <td class="num">${num0(w.crews_assigned)}</td>
      <td>${w.critical ? critBadge(t("alloc.critical")) : `<span class="badge">${esc(t("alloc.ok"))}</span>`}</td>
    </tr>`;
  }).join("");
  return `<div class="table-wrap"><table class="rank">
    <thead><tr><th scope="col">${esc(t("wards.rank"))}</th><th scope="col">${esc(t("wards.ward"))}</th><th scope="col">${esc(t("wards.tier"))}</th>
      <th scope="col" class="num">${esc(t("alloc.density"))}</th><th scope="col" class="num">${esc(t("alloc.forecast"))}</th>
      ${ITEMS.map((k) => `<th scope="col" class="num">${esc(t(`alloc.col_${k}`))}</th>`).join("")}
      <th scope="col">${esc(t("alloc.fulfilled"))}</th><th scope="col" class="num">${esc(t("alloc.col_crews"))}</th><th scope="col">${esc(t("alloc.status"))}</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function orderPlan(plan, s) {
  const key = s.basis === "range_high" ? "cautious" : "expected";
  const rows = plan.weeks.map((w) => `<tr><td><b>${esc(t("outlook.plus_weeks", { h: num0(w.horizon) }))}</b><span class="city">${esc(dateLong(w.forecast_week))}</span></td>
    ${ITEMS.map((k) => `<td class="num">${num0(w[key][k])}</td>`).join("")}</tr>`).join("");
  const o = plan.order[key];
  return `<div class="warn-head"><h2>${esc(t("alloc.plan_title"))}</h2><span>${esc(t(`alloc.basis_${key === "cautious" ? "range_high" : "predicted_cases"}`))}${plan.city ? ` · ${esc(cityName(plan.city))}` : ""}</span></div>
  <div class="table-wrap"><table class="rank">
    <thead><tr><th scope="col">${esc(t("alloc.week"))}</th>${ITEMS.map((k) => `<th scope="col" class="num">${esc(t(`alloc.col_${k}`))}</th>`).join("")}</tr></thead>
    <tbody>${rows}<tr><td><b>${esc(t("alloc.order_now"))}</b></td><td class="num c">${num0(o.saline)}</td><td class="num c">${num0(o.kits)}</td><td class="num c">${num0(o.beds_peak)}</td></tr></tbody>
  </table></div>
  <p class="acc-note">${esc(t("alloc.plan_rule"))}</p>`;
}

function how(a) {
  const whys = [a.priority.tier_weight, a.priority.fill_rule, a.stock.full_stock_basis, a.crews.assignment_rule].filter(Boolean);
  return `<details class="why" open><summary>${esc(t("alloc.how"))}</summary><ul lang="en">${whys.map((x) => `<li>${esc(x.why)}</li>`).join("")}</ul></details>`;
}

function csv(r) {
  const head = ["priority_rank", "ward_id", "city", "risk_tier", "population_density_sqkm", "priority_score", "predicted_cases", "demand_cases",
    ...ITEMS.flatMap((k) => [`${k}_required`, `${k}_allocated`, `${k}_deficit`]), "fulfilled_pct", "crews_assigned", "critical"];
  downloadCsv(`dengue_allocation_${r.forecast_week}_h${r.horizon}_stock${r.stock_pct}.csv`, head, r.wards.map((w) => head.map((k) => w[k])));
}
