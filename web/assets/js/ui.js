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
