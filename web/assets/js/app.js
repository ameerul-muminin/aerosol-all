// App shell: state in the URL, hash router (#outlook, #wards, #accuracy, #cost, #allocation, #overflow, #simulator,
// #dispatch), sidebar inputs from /api/meta, EN/বাংলা toggle.
// URL shape: /app/?ward=CCC_Ward_01&as_of=2025-07-14&h=3&city=Gazipur&stock=50&crews=20&beds=80&basis=range_high#allocation
// stock/crews/beds/basis are shared by the four Phase 2 views, so Dispatch uses what was set on Allocation and Overflow.
import { api } from "./api.js";
import { initI18n, setLang, getLang } from "./i18n.js";
import { showError } from "./ui.js";
import { cityName, wardName, dateLong, esc } from "./format.js";
import * as outlook from "./views/outlook.js";
import * as wards from "./views/wards.js";
import * as accuracy from "./views/accuracy.js";
import * as cost from "./views/cost.js";
import * as allocation from "./views/allocation.js";
import * as overflow from "./views/overflow.js";
import * as simulator from "./views/simulator.js";
import * as dispatch from "./views/dispatch.js";

const VIEWS = { outlook, wards, accuracy, cost, allocation, overflow, simulator, dispatch };
// views about the whole region, not one ward: the sidebar Ward field is hidden on these
const REGION_VIEWS = ["wards", "cost", "allocation", "overflow", "simulator", "dispatch"];
const STOCK_DEFAULT = 60;
const whole = (v) => (/^\d+$/.test(v || "") ? v : "");   // "" = let the server use its default
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
    stock: Number(whole(q.get("stock")) || STOCK_DEFAULT),
    crews: whole(q.get("crews")),
    beds: whole(q.get("beds")),
    basis: q.get("basis") === "range_high" ? "range_high" : "",
  };
}

function writeUrl() {
  const q = new URLSearchParams({ ward: state.ward, as_of: state.as_of, h: state.h });
  if (state.city) q.set("city", state.city);
  if (state.stock !== STOCK_DEFAULT) q.set("stock", state.stock);
  if (state.crews !== "") q.set("crews", state.crews);
  if (state.beds !== "") q.set("beds", state.beds);
  if (state.basis) q.set("basis", state.basis);
  history.replaceState(null, "", `?${q}#${state.view}`);
}

/** Views call ctx.set({...}) to change state; it updates the URL and re-renders without re-animating. */
function set(patch) {
  state = { ...state, ...patch };
  writeUrl();
  syncSidebar();
  render();
}

/** Live controls call ctx.save({...}): it updates the state and the URL but does not re-render, so a slider
 *  keeps focus while it is dragged. The view repaints its own results. */
function save(patch) {
  state = { ...state, ...patch };
  writeUrl();
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
  $("f-ward").hidden = REGION_VIEWS.includes(state.view);   // views that are not about one ward
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.view === state.view) {
      a.setAttribute("aria-current", "page");
      const row = a.parentElement;                 // phones: keep the current tab visible in the scrolling row
      if (row.scrollWidth > row.clientWidth) row.scrollLeft = Math.max(0, a.offsetLeft - 16);
    } else a.removeAttribute("aria-current");
  });
  document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === getLang())));
}

function render() {
  const key = state.view;
  const animate = rendered !== key;
  rendered = key;
  main.classList.toggle("anim", animate);
  document.body.classList.toggle("no-anim", animate);   // thumbs jump into place on first paint
  VIEWS[state.view].render(main, { state, meta, set, save, retry: render });
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
