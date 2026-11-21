// App shell: state in the URL, hash router (#outlook, #wards), sidebar inputs from /api/meta, EN/বাংলা toggle.
// URL shape: /app/?ward=CCC_Ward_01&as_of=2025-07-14&h=3&city=Gazipur#wards
import { api } from "./api.js";
import { initI18n, setLang, getLang } from "./i18n.js";
import { showError } from "./ui.js";
import { cityName, wardName, dateLong, esc } from "./format.js";
import * as outlook from "./views/outlook.js";
import * as wards from "./views/wards.js";

const VIEWS = { outlook, wards };
const $ = (id) => document.getElementById(id);
const main = $("main");
let meta;
let state;
let rendered = "";   // view + state key last rendered (bars animate only when the view itself changes)

function readState() {
  const q = new URLSearchParams(location.search);
  const firstWard = meta.cities[0].wards[0];
  const allWards = meta.cities.flatMap((c) => c.wards);
  const h = Number(q.get("h"));
  return {
    view: VIEWS[location.hash.slice(1)] ? location.hash.slice(1) : "outlook",
    ward: allWards.includes(q.get("ward")) ? q.get("ward") : firstWard,
    as_of: q.get("as_of") || meta.as_of.default,
    h: meta.horizons.includes(h) ? h : 3,
    city: meta.cities.some((c) => c.city === q.get("city")) ? q.get("city") : "",
  };
}

function writeUrl() {
  const q = new URLSearchParams({ ward: state.ward, as_of: state.as_of, h: state.h });
  if (state.city) q.set("city", state.city);
  history.replaceState(null, "", `?${q}#${state.view}`);
}

/** Views call ctx.set({...}) to change state; it updates the URL and re-renders without re-animating. */
function set(patch) {
  state = { ...state, ...patch };
  writeUrl();
  syncSidebar();
  render();
}

function fillSidebar() {
  $("ward").innerHTML = meta.cities.map((c) =>
    `<optgroup label="${esc(cityName(c.city))}">${c.wards.map((w) => `<option value="${w}">${esc(wardName(w))}</option>`).join("")}</optgroup>`).join("");
  const weeks = [];
  for (let d = new Date(`${meta.as_of.max}T00:00:00Z`); d >= new Date(`${meta.as_of.min}T00:00:00Z`); d.setUTCDate(d.getUTCDate() - 7)) {
    weeks.push(d.toISOString().slice(0, 10));
  }
  $("asof").innerHTML = weeks.map((w) => `<option value="${w}">${esc(dateLong(w))}</option>`).join("");
  syncSidebar();
}

function syncSidebar() {
  $("ward").value = state.ward;
  $("asof").value = state.as_of;
  $("f-ward").hidden = state.view !== "outlook";
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.view === state.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === getLang())));
}

function render() {
  const key = state.view;
  const animate = rendered !== key;
  rendered = key;
  main.classList.toggle("anim", animate);
  document.body.classList.toggle("no-anim", animate);   // thumbs jump into place on first paint
  VIEWS[state.view].render(main, { state, meta, set, retry: render });
}

async function boot() {
  await initI18n();
  try {
    meta = await api.meta();
  } catch (err) {
    showError(main, err, () => location.reload());
    return;
  }
  state = readState();
  writeUrl();
  fillSidebar();

  $("ward").onchange = (e) => set({ ward: e.target.value });
  $("asof").onchange = (e) => set({ as_of: e.target.value });
  document.querySelectorAll(".lang button").forEach((b) => {
    b.onclick = async () => {
      if (b.dataset.lang === getLang()) return;
      await setLang(b.dataset.lang);
      fillSidebar();
      render();
    };
  });
  window.addEventListener("hashchange", () => {
    state = { ...state, view: VIEWS[location.hash.slice(1)] ? location.hash.slice(1) : "outlook" };
    writeUrl();
    syncSidebar();
    render();
    main.focus({ preventScroll: true });
  });
  render();
}

boot();
