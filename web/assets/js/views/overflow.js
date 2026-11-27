// O4 Hospital overflow: beds-free slider, per-city utilisation and headroom, transfers (always inside one city),
// ward deficits. Data: GET /api/overflow. Every number comes from the answer.
import { api } from "../api.js";
import { t } from "../i18n.js";
import { num0, num1, dateLong, cityName, wardName, esc } from "../format.js";
import { showError, ticket, segment, debounce, loadingHTML, rangeField, alertIcon, inlineError } from "../ui.js";

let showAllWards = false;

const fetchFor = (s) => api.overflow(s.as_of, s.h, s.beds, s.basis);

export async function render(root, { state, meta, save, retry }) {
  const s = { ...state };
  const cur = ticket(root);
  root.innerHTML = loadingHTML(t("flow.title"));
  let r;
  try {
    r = await fetchFor(s);
  } catch (err) {
    if (cur()) showError(root, err, retry);
    return;
  }
  if (!cur()) return;

  const bc = r.assumptions.bed_capacity.beds_available_pct;
  const [bmin, bmax] = bc.range;
  const pct = (v) => `${num0(v)}%`;
  const beds = s.beds !== "" ? Number(s.beds) : r.beds_available_pct;
  root.innerHTML = `
  <div class="head"><h1 class="display">${esc(t("flow.title"))}</h1><p id="o-sub"></p></div>
  <div class="toolbar">
    <div class="field"><span id="oh-l">${esc(t("outlook.horizon_label"))}</span>
      <div class="seg" id="o-h" role="group" aria-labelledby="oh-l"><span class="thumb"></span>
        ${meta.horizons.map((h) => `<button type="button" data-v="${h}">${esc(t("outlook.plus_weeks", { h: num0(h) }))}</button>`).join("")}</div></div>
    ${rangeField("o-beds", t("flow.beds_free"), { min: bmin, max: bmax, step: 5, value: beds, fmt: pct })}
    <div class="field"><span id="ob-l">${esc(t("alloc.demand"))}</span>
      <div class="seg" id="o-basis" role="group" aria-labelledby="ob-l"><span class="thumb"></span>
        ${r.assumptions.demand.basis.allowed.map((b) => `<button type="button" data-v="${b}">${esc(t(`alloc.basis_${b}`))}</button>`).join("")}</div></div>
  </div>
  <p class="inline-error" role="alert" id="o-err" hidden></p>
  <div class="out" id="o-out"></div>`;

  const out = root.querySelector("#o-out");
  const err = root.querySelector("#o-err");
  paint(out, r);

  async function load() {
    const mine = ticket(out);
    out.setAttribute("aria-busy", "true");
    try {
      const r2 = await fetchFor(s);
      if (!mine() || !out.isConnected) return;
      inlineError(err, null);
      paint(out, r2);
    } catch (e) {
      if (mine()) inlineError(err, e);
    } finally {
      if (mine()) out.removeAttribute("aria-busy");
    }
  }
  const change = (patch) => { Object.assign(s, patch); save(patch); load(); };
  const later = debounce(change, 200);

  segment(root.querySelector("#o-h"), s.h, (v) => change({ h: Number(v) }));
  segment(root.querySelector("#o-basis"), s.basis || "predicted_cases", (v) => change({ basis: v === "predicted_cases" ? "" : v }));
  const slider = root.querySelector("#o-beds");
  slider.oninput = () => {
    root.querySelector("#o-beds-out").textContent = pct(slider.value);
    later({ beds: slider.value });
  };
}

function paint(out, r) {
  out.closest(".main").querySelector("#o-sub").textContent =
    t("alloc.subtitle", { week: dateLong(r.forecast_week), h: num0(r.horizon) });

  const stuck = r.cities.filter((c) => c.unroutable > 0);
  const moved = r.transfers.reduce((n, x) => n + x.patients, 0);
  const alerts = stuck.map((c) => `<div class="alert" role="alert">${alertIcon}<span>${esc(t("flow.unroutable_alert", { n: num0(c.unroutable), city: cityName(c.city) }))}</span></div>`).join("");
  const calm = moved > 0 ? t("flow.moved", { n: num0(moved) }) : (stuck.length ? "" : t("flow.none_over"));

  const cities = r.cities.map((c) => {
    const u = c.utilisation_pct;
    const over = u > 100;
    return `<section class="card citycard span-4"><h2>${esc(cityName(c.city))}</h2>
      <div class="util"><span>${esc(t("flow.used"))}</span><b class="${over ? "deficit" : ""}">${num1(u)}%</b></div>
      <span class="hbar" role="img" aria-label="${esc(t("flow.util_aria", { city: cityName(c.city), pct: num1(u) }))}"><i class="${over ? "over" : ""}" style="transform:scaleX(${Math.min(u, 100) / 100})"></i></span>
      <p>${esc(t("flow.needed", { d: num0(c.demand), a: num0(c.available_capacity), f: num0(c.full_capacity) }))}</p>
      <p class="head-room">${esc(c.headroom === 1 ? t("flow.headroom_one") : t("flow.headroom", { n: num0(c.headroom) }))}</p>
      ${c.unroutable > 0 ? `<p class="deficit">${alertIcon} ${esc(t("flow.unroutable", { n: num0(c.unroutable) }))}</p>` : ""}
    </section>`;
  }).join("");

  const xfers = r.transfers.length
    ? `<ol class="xfer">${r.transfers.map((x) => `<li><b>${esc(wardName(x.from_ward))} → ${esc(wardName(x.to_ward))}</b>
        <span>${esc(cityName(x.city))} · ${esc(x.patients === 1 ? t("flow.patient") : t("flow.patients", { n: num0(x.patients) }))}</span></li>`).join("")}</ol>`
    : `<p class="warn-none">${esc(t("flow.no_transfers"))}</p>`;

  out.innerHTML = `${alerts}${calm ? `<p class="calm">${esc(calm)}</p>` : ""}
  <div class="grid">
    ${cities}
    <section class="card span-6"><h2>${esc(t("flow.transfers"))}</h2>${xfers}</section>
    <section class="card span-6" id="o-wards">${wardTable(r)}</section>
    <section class="card span-12"><p class="acc-note" style="margin:0" lang="en"><b>${esc(t("flow.same_city"))}</b> ${esc(r.assumptions.overflow.same_city_only.why)}</p></section>
  </div>`;
  bindToggle(out, r);
}

function wardTable(r) {
  const busy = (w) => w.routed_out > 0 || w.routed_in > 0 || w.net_deficit > 0;
  const list = showAllWards ? r.wards : r.wards.filter(busy);
  const rows = list.map((w) => `<tr>
    <td><b>${esc(wardName(w.ward_id))}</b></td><td>${esc(cityName(w.city))}</td>
    <td class="num">${num0(w.beds_required)}</td><td class="num">${num0(w.available_capacity)} / ${num0(w.full_capacity)}</td>
    <td class="num">${num0(w.routed_out)}</td><td class="num">${num0(w.routed_in)}</td>
    <td class="num${w.net_deficit > 0 ? " deficit" : ""}">${num0(w.net_deficit)}</td></tr>`).join("");
  return `<div class="warn-head"><h2>${esc(t("flow.wards_title"))}</h2></div>
  ${list.length ? `<div class="table-wrap"><table class="rank">
    <thead><tr><th scope="col">${esc(t("wards.ward"))}</th><th scope="col">${esc(t("ctl.city"))}</th><th scope="col" class="num">${esc(t("flow.col_needed"))}</th>
      <th scope="col" class="num">${esc(t("flow.col_capacity"))}</th><th scope="col" class="num">${esc(t("flow.col_out"))}</th><th scope="col" class="num">${esc(t("flow.col_in"))}</th>
      <th scope="col" class="num">${esc(t("flow.col_deficit"))}</th></tr></thead><tbody>${rows}</tbody></table></div>`
    : `<p class="warn-none">${esc(t("flow.no_busy_wards"))}</p>`}
  <div class="more-row"><button type="button" class="btn small" id="o-all" aria-pressed="${showAllWards}">${esc(showAllWards ? t("flow.show_busy") : t("flow.show_all"))}</button></div>`;
}

function bindToggle(out, r) {
  out.querySelector("#o-all").onclick = () => {
    showAllWards = !showAllWards;
    const card = out.querySelector("#o-wards");
    card.innerHTML = wardTable(r);
    bindToggle(out, r);
    card.querySelector("#o-all").focus();
  };
}
