// D3 Dispatch: ward notices (SMS + WhatsApp, EN or বাংলা) built from the same settings as Supply allocation
// and Hospital overflow (shared URL state), copy buttons, printable brief / save as PDF.
// Data: GET /api/dispatch and the HTML page /api/dispatch/brief. Notice text is written by the server (templates, no LLM).
import { api } from "../api.js";
import { t, getLang } from "../i18n.js";
import { num0, num1, dateLong, cityName, wardName, tierHTML, esc } from "../format.js";
import { showError, ticket, segment, loadingHTML, alertIcon, critBadge, inlineError, copyText } from "../ui.js";

let noticeLang = "";     // "" = follow the UI language
let criticalOnly = false;

const query = (s) => ({
  as_of: s.as_of, horizon: s.h, stock_pct: s.stock, crews: s.crews, beds_available_pct: s.beds,
  lang: noticeLang || getLang(), city: s.city,
});

export async function render(root, { state, meta, save, retry }) {
  const s = { ...state };
  const cur = ticket(root);
  root.innerHTML = loadingHTML(t("disp.title"));
  let r;
  try {
    r = await api.dispatch(query(s));
  } catch (err) {
    if (cur()) showError(root, err, retry);
    return;
  }
  if (!cur()) return;

  root.innerHTML = `
  <div class="head"><h1 class="display">${esc(t("disp.title"))}</h1><p id="d-sub"></p></div>
  <div class="settings-line" id="d-set"></div>
  <div class="toolbar">
    <div class="field"><span id="dl-l">${esc(t("disp.lang"))}</span>
      <div class="seg" id="d-lang" role="group" aria-labelledby="dl-l"><span class="thumb"></span>
        <button type="button" data-v="en" lang="en">English</button><button type="button" data-v="bn" lang="bn">বাংলা</button></div></div>
    <label class="field"><span>${esc(t("ctl.city"))}</span><select id="d-city">
      <option value="">${esc(t("ctl.city_all"))}</option>
      ${meta.cities.map((c) => `<option value="${esc(c.city)}"${c.city === s.city ? " selected" : ""}>${esc(cityName(c.city))}</option>`).join("")}
    </select></label>
    <label class="check"><input type="checkbox" id="d-crit"${criticalOnly ? " checked" : ""}>${esc(t("disp.critical_only"))}</label>
  </div>
  <p class="inline-error" role="alert" id="d-err" hidden></p>
  <p class="sr-only" aria-live="polite" id="d-live"></p>
  <div class="out" id="d-out"></div>`;

  const out = root.querySelector("#d-out");
  const err = root.querySelector("#d-err");
  const live = root.querySelector("#d-live");
  let cur2 = r;
  paint(root, out, r, s, live, err);

  async function load() {
    const mine = ticket(out);
    out.setAttribute("aria-busy", "true");
    try {
      const r2 = await api.dispatch(query(s));
      if (!mine() || !out.isConnected) return;
      inlineError(err, null);
      cur2 = r2;
      paint(root, out, r2, s, live, err);
    } catch (e) {
      if (mine()) inlineError(err, e);
    } finally {
      if (mine()) out.removeAttribute("aria-busy");
    }
  }

  segment(root.querySelector("#d-lang"), noticeLang || getLang(), (v) => { noticeLang = v; load(); });
  root.querySelector("#d-city").onchange = (e) => { s.city = e.target.value; save({ city: s.city }); load(); };
  root.querySelector("#d-crit").onchange = (e) => { criticalOnly = e.target.checked; paint(root, out, cur2, s, live, err); };
}

function paint(root, out, r, s, live, err) {
  const sum = r.summary;
  root.querySelector("#d-sub").textContent = t("alloc.subtitle", { week: dateLong(r.forecast_week), h: num0(r.horizon) });
  root.querySelector("#d-set").innerHTML = `<span>${esc(t("disp.settings_label"))}</span>
    <b>${esc(t("disp.settings", { stock: num0(r.stock_pct), crews: num0(r.crews), beds: num0(r.beds_available_pct), h: num0(r.horizon) }))}</b>
    <a href="#allocation">${esc(t("disp.change"))}</a> <a href="#overflow">${esc(t("disp.change_beds"))}</a>`;

  const list = criticalOnly ? r.notices.filter((n) => n.critical) : r.notices;
  const nl = r.lang;
  const cards = list.map((n, i) => {
    const sup = ["saline", "kits", "beds"].map((k) => {
      const d = n[`${k}_deficit`];
      return `<span>${esc(t(`alloc.col_${k}`))} <b class="${d > 0 ? "deficit" : ""}">${num0(n[`${k}_allocated`])}/${num0(n[`${k}_required`])}</b>${d > 0 ? ` <span class="deficit">${esc(t("alloc.short", { n: num0(d) }))}</span>` : ""}</span>`;
    }).join("");
    const o = n.overflow || {};
    return `<section class="card span-12 notice">
      <div class="notice-head"><h2>${esc(wardName(n.ward_id))}</h2><span class="city">${esc(cityName(n.city))} · #${num0(n.priority_rank)}</span>
        ${tierHTML(n.risk_tier)}${n.critical ? critBadge(t("alloc.critical")) : ""}</div>
      <div class="facts">
        <span>${esc(t("disp.forecast"))} <b>${esc(t("warn.cases", { n: num1(n.predicted_cases) }))}</b> (${num1(n.range_low)}–${num1(n.range_high)})</span>
        ${sup}
        <span>${esc(t("alloc.col_crews"))} <b>${num0(n.crews_assigned)}</b></span>
        <span>${esc(t("disp.beds_inout"))} <b>${num0(o.routed_in || 0)} / ${num0(o.routed_out || 0)}</b></span>
        ${n.unroutable > 0 ? `<span class="deficit">${alertIcon} ${esc(t("flow.unroutable", { n: num0(n.unroutable) }))}</span>` : ""}
      </div>
      <div class="msgs" lang="${esc(nl)}">
        <div class="msg"><span class="k" lang="${esc(getLang())}">${esc(t("disp.sms_meta", { c: num0(n.sms_chars), s: num0(n.sms_segments) }))}
          <button type="button" class="btn small" data-copy="sms" data-i="${i}">${esc(t("disp.copy"))}</button></span><pre>${esc(n.sms)}</pre></div>
        <div class="msg"><span class="k" lang="${esc(getLang())}">${esc(t("disp.whatsapp"))}
          <button type="button" class="btn small" data-copy="whatsapp" data-i="${i}">${esc(t("disp.copy"))}</button></span><pre class="wa">${esc(n.whatsapp)}</pre></div>
      </div></section>`;
  }).join("");

  out.innerHTML = `
  <div class="grid">
    <section class="card tile span-3"><span class="k">${esc(t("disp.wards"))}</span><b>${num0(sum.wards)}</b></section>
    <section class="card tile span-3"><span class="k">${esc(t("disp.critical"))}</span><b class="${sum.critical ? "deficit" : ""}">${num0(sum.critical)}</b>
      ${sum.critical ? `<span class="sub deficit">${alertIcon} ${esc(t("disp.critical_note"))}</span>` : ""}</section>
    <section class="card tile span-3"><span class="k">${esc(t("disp.unroutable"))}</span><b class="${sum.unroutable_patients ? "deficit" : ""}">${num0(sum.unroutable_patients)}</b></section>
    <section class="card tile span-3"><span class="k">${esc(t("disp.longest"))}</span><b>${esc(t("disp.n_sms", { n: num0(sum.max_sms_segments) }))}</b></section>
  </div>
  <div class="btnrow" style="margin-top:16px">
    <button type="button" class="btn primary" id="d-brief">${esc(t("disp.open_brief"))}</button>
    <button type="button" class="btn" id="d-print">${esc(t("disp.print"))}</button>
    <button type="button" class="btn" id="d-all" data-copy="all">${esc(t("disp.copy_all", { n: num0(list.length) }))}</button>
  </div>
  <div class="grid">${cards || `<section class="card span-12"><p class="warn-none">${esc(t("disp.none"))}</p></section>`}</div>`;

  const url = api.briefUrl(query(s));
  out.querySelector("#d-brief").onclick = () => {
    if (!window.open(url, "_blank")) inlineError(err, new Error(t("disp.popup_blocked")));
  };
  out.querySelector("#d-print").onclick = () => {
    const w = window.open(url, "_blank");
    if (!w) { inlineError(err, new Error(t("disp.popup_blocked"))); return; }
    w.addEventListener("load", () => { w.focus(); w.print(); }, { once: true });
  };
  out.querySelectorAll("[data-copy]").forEach((b) => (b.onclick = async () => {
    const text = b.dataset.copy === "all" ? list.map((n) => n.sms).join("\n\n") : list[Number(b.dataset.i)][b.dataset.copy];
    const ok = await copyText(text);
    const msg = ok ? t("disp.copied") : t("disp.copy_failed");
    live.textContent = "";
    live.textContent = msg;
    const was = b.textContent;
    b.textContent = msg;
    setTimeout(() => { if (b.isConnected) b.textContent = was; }, 1500);
  }));
}
