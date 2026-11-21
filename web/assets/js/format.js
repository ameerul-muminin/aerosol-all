// Formatting in the active language: numbers (Bangla digits in বাংলা), dates, ward/city/tier/driver names.
import { t, has, getLang } from "./i18n.js";

const locale = () => (getLang() === "bn" ? "bn-BD" : "en-GB");
const nf = (max, min = 0) => new Intl.NumberFormat(locale(), { maximumFractionDigits: max, minimumFractionDigits: min });

export const num0 = (n) => nf(0).format(n);
export const num1 = (n) => nf(1).format(n);
export const num2 = (n) => nf(2).format(n);
/** signed, with a real minus sign: +3.8 / −0.4 */
export const signed1 = (n) => `${n >= 0 ? "+" : "−"}${num1(Math.abs(n))}`;

const asDate = (iso) => new Date(`${iso}T00:00:00`);
export const dateLong = (iso) => asDate(iso).toLocaleDateString(locale(), { day: "numeric", month: "short", year: "numeric" });
export const dateShort = (iso) => asDate(iso).toLocaleDateString(locale(), { day: "numeric", month: "short" });

export const cityName = (city) => t(`city.${city}`, {}, city);
export const wardNo = (id) => Number(id.split("_Ward_")[1]);
export const wardName = (id) => t("ward.name", { n: num0(wardNo(id)) });
export const wardFull = (id, city) => t("ward.full", { city: cityName(city), n: num0(wardNo(id)) });

export const LEVEL = { Low: 1, Medium: 2, High: 3, Critical: 4 };
export const tierName = (tier) => t(`tier.${tier}`, {}, tier);
const glyph = (tier) => `<span class="tg" data-l="${LEVEL[tier] || 0}" aria-hidden="true"><i></i><i></i><i></i><i></i></span>`;
/** Risk tier = colour + 4-bar glyph + word. `risk: true` -> "Critical risk". */
export function tierHTML(tier, { risk = false } = {}) {
  const word = risk ? t("tier.risk", { tier: tierName(tier) }) : tierName(tier);
  return `<span class="tier t-${tier}">${glyph(tier)}${esc(word)}</span>`;
}

/** Driver labels come from the API in English; translate them here (fallback: the API text). */
export const driverName = (label) => t(`driver.${label}`, {}, label);
export const driverDetail = (label) => (has(`driver.${label}.detail`) ? t(`driver.${label}.detail`) : "");

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
