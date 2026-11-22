// Marketing page: live chips from /api (nothing typed into the HTML) + EN/বাংলা toggle.
// Chips: Critical and High ward counts and the highest ward from /api/ranking, the test-period error
// against the seasonal baseline and the per-horizon accuracy cards from /api/metrics. The only hard-coded numbers are the three brief facts.
import { initI18n, setLang, getLang, t } from "./i18n.js";
import { api, get } from "./api.js";
import { num0, num1, num2, dateLong, wardFull, tierHTML, esc } from "./format.js";

const HORIZON = 3;   // the middle of the 2–4 week window
const $ = (id) => document.getElementById(id);
let data = null;     // { meta, rank, m }

function chip(cls, k, v, s) {
  return `<div class="chip ${cls}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${esc(s)}</div></div>`;
}

/** "2024-01..2025-12" -> "2024–25" (Bangla digits in বাংলা) */
function testYears(split) {
  const y = (n) => n.toLocaleString(getLang() === "bn" ? "bn-BD" : "en-GB", { useGrouping: false });
  const [a, b] = split.test.split("..").map((s) => Number(s.slice(0, 4)));
  return a === b ? y(a) : `${y(a)}–${y(b % 100).padStart(2, y(0))}`;
}

function render() {
  if (!data) return;
  const { meta, rank, m } = data;
  const tc = rank.tier_counts, top = rank.wards[0];
  const h = m.horizons.find((x) => x.horizon === HORIZON);
  const of = t("landing.wards_of", { n: num0(meta.ward_count) });
  const cut = Math.round((1 - h.mae / h.baseline_mae) * 100);

  $("sub").textContent = t("landing.sub", { n: num0(meta.ward_count), a: num0(Math.min(...meta.horizons)), b: num0(Math.max(...meta.horizons)) });
  $("live").textContent = t("landing.live", { date: dateLong(rank.forecast_week), h: num0(HORIZON) });
  $("chips").innerHTML = [
    chip("c1", tierHTML("Critical"), num0(tc.Critical), of),
    chip("c2", tierHTML("High"), num0(tc.High), of),
    chip("c3", esc(t("landing.chip_top")), num1(top.predicted_cases), t("landing.chip_top_s", { ward: wardFull(top.ward_id, top.city) })),
    chip("c4", esc(t("landing.chip_err")), `${cut >= 0 ? "−" : "+"}${num0(Math.abs(cut))}%`, t("landing.chip_err_s", { years: testYears(m.split) })),
  ].join("");
  renderAccuracy(m);
}

/** One card per horizon: model MAE vs the seasonal baseline (bars on one shared scale), RMSE, 80% coverage. */
function renderAccuracy(m) {
  const years = testYears(m.split);
  const top = Math.max(...m.horizons.flatMap((h) => [h.mae, h.baseline_mae]));
  const bar = (cls, label, v) =>
    `<div class="bar ${cls}"><span>${esc(label)}</span><i style="--w:${(v / top).toFixed(3)}"></i><b>${num2(v)}</b></div>`;
  $("acc-sub").textContent = t("landing.acc_sub", { years });
  $("acc-grid").innerHTML = m.horizons.map((h) => `
    <article class="acc-card">
      <h3>${esc(t("landing.acc_h", { h: num0(h.horizon) }))}</h3>
      <div class="big">${num2(h.mae)}</div>
      <p class="unit">${esc(t("landing.acc_unit"))}</p>
      <div class="bars" role="img" aria-label="${esc(`${t("landing.acc_model")} ${num2(h.mae)}, ${t("landing.acc_base")} ${num2(h.baseline_mae)}`)}">
        ${bar("model", t("landing.acc_model"), h.mae)}${bar("base", t("landing.acc_base"), h.baseline_mae)}
      </div>
      <p class="foot">${esc(t("landing.acc_rmse", { rmse: num2(h.rmse), cov: num0(h.interval80_coverage * 100) }))}</p>
    </article>`).join("");
}

function markLang() {
  document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === getLang())));
}

async function main() {
  await initI18n();
  markLang();
  document.querySelectorAll(".lang button").forEach((b) => {
    b.onclick = async () => { await setLang(b.dataset.lang); markLang(); render(); };
  });
  try {
    const meta = await api.meta();
    const [rank, m] = await Promise.all([api.ranking(meta.as_of.default, HORIZON), get("/api/metrics")]);
    data = { meta, rank, m };
    render();
  } catch (e) {
    $("chips").hidden = true;
    $("sub").textContent = "";
    $("err").textContent = e.message;
    $("err").hidden = false;
  }
}

main();
