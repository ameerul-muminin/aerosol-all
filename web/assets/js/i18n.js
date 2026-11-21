// EN / বাংলা strings. Every visible string goes through t(). The choice is saved in localStorage.
// Keys are "section.key"; only the FIRST dot splits, so API labels with dots/brackets can be keys
// (e.g. t("driver.Rainfall (lags 0-8 weeks).detail")).
export const LANGS = ["en", "bn"];
const STORE_KEY = "aerosol-all.lang";
let lang = "en";
let dict = {};
const loaded = {};

function saved() {
  try { return localStorage.getItem(STORE_KEY); } catch { return null; }
}

export async function initI18n() {
  const want = saved();
  await setLang(LANGS.includes(want) ? want : "en", { persist: false });
}

export async function setLang(next, { persist = true } = {}) {
  if (!loaded[next]) {
    const res = await fetch(`/assets/i18n/${next}.json`);
    loaded[next] = await res.json();
  }
  lang = next;
  dict = loaded[next];
  document.documentElement.lang = next;
  if (persist) { try { localStorage.setItem(STORE_KEY, next); } catch { /* private mode */ } }
  applyStatic();
}

export const getLang = () => lang;

/** t("outlook.range", {low: 7, high: 20}); a missing key returns `fallback` (or the key) so nothing crashes. */
export function t(key, vars = {}, fallback) {
  const i = key.indexOf(".");
  const section = dict[key.slice(0, i)];
  let s = section ? section[key.slice(i + 1)] : undefined;
  if (s === undefined) s = fallback !== undefined ? fallback : key;
  return s.replace(/\{(\w+)\}/g, (_, k) => (vars[k] !== undefined ? vars[k] : `{${k}}`));
}

export const has = (key) => {
  const i = key.indexOf(".");
  return Boolean(dict[key.slice(0, i)] && dict[key.slice(0, i)][key.slice(i + 1)] !== undefined);
};

/** Fill static markup: <span data-i18n="nav.outlook">, <x data-i18n-attr="aria-label:nav.label"> */
export function applyStatic(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll("[data-i18n-attr]").forEach((el) => {
    el.dataset.i18nAttr.split(";").forEach((pair) => {
      const [attr, key] = pair.split(":");
      el.setAttribute(attr.trim(), t(key.trim()));
    });
  });
  document.title = t("app.title");
}
