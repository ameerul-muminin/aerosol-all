// Small shared UI helpers (no data, no strings of their own).
import { t } from "./i18n.js";
import { esc } from "./format.js";

/** Error panel with a retry button. Shows the API's friendly message, never a stack trace. */
export function showError(root, err, retry) {
  root.innerHTML = `<div class="state" role="alert"><h2>${esc(t("state.error_title"))}</h2><p>${esc(err.message)}</p>
    <button class="btn" type="button">${esc(t("state.retry"))}</button></div>`;
  root.querySelector("button").onclick = retry;
}

/** Guards against out-of-order responses: only the latest render of a root may paint. */
export function ticket(root) {
  const id = String((Number(root.dataset.req) || 0) + 1);
  root.dataset.req = id;
  return () => root.dataset.req === id;
}

/** Segmented control: one sliding thumb under the pressed button. */
export function segment(el, value, onPick) {
  const btns = [...el.querySelectorAll("button")];
  const thumb = el.querySelector(".thumb");
  const place = (v) => {
    btns.forEach((b, i) => {
      const on = b.dataset.v === String(v);
      b.setAttribute("aria-pressed", String(on));
      if (on) { thumb.style.width = `calc((100% - 6px) / ${btns.length})`; thumb.style.transform = `translateX(${i * 100}%)`; }
    });
  };
  place(value);
  btns.forEach((b) => (b.onclick = () => { place(b.dataset.v); onPick(b.dataset.v); }));
}

/** Run `fn` only after the input has been quiet for `ms` (live sliders: one request per pause, not per pixel). */
export function debounce(fn, ms = 200) {
  let id;
  return (...args) => { clearTimeout(id); id = setTimeout(() => fn(...args), ms); };
}

/** Loading shell for a view's first paint: the title, then skeletons (never fake numbers). */
export function loadingHTML(title) {
  return `<div class="head"><h1 class="display">${esc(title)}</h1><p><span class="skel">${esc(t("state.loading"))}</span></p></div>
    <div class="grid" aria-busy="true"><section class="card span-12"><span class="skel" style="height:320px;width:100%"></span></section></div>`;
}

/** Range slider field: label, live value and a 44 px track. `fmt` turns the value into the shown text. */
export function rangeField(id, label, { min, max, step, value, fmt }) {
  return `<div class="field range-field"><span class="rf-top"><label for="${id}">${esc(label)}</label>
      <output for="${id}" id="${id}-out">${esc(fmt(value))}</output></span>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"></div>`;
}

/** Warning triangle for alert banners and badges (always shown next to words, never colour alone). */
export const alertIcon = `<svg class="ico" viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M8 1.5 15 14H1Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 6v3.6M8 11.4v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`;

/** Red "Critical" style badge: icon + word. */
export const critBadge = (word) => `<span class="badge-crit">${alertIcon}${esc(word)}</span>`;

/** Show an API error under the controls (live changes), or clear it with `null`. */
export function inlineError(el, err) {
  el.hidden = !err;
  el.textContent = err ? err.message : "";
}

/** Download rows as a CSV file. */
export function downloadCsv(filename, head, rows) {
  const cell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  const text = [head, ...rows].map((r) => r.map(cell).join(",")).join("\n");
  const blob = new Blob([`${text}\n`], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: filename });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** Copy text to the clipboard; falls back to select + execCommand where the Clipboard API is missing. */
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.append(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    ta.remove();
    return ok;
  }
}
