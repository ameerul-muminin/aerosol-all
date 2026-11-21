// PROTOTYPE ONLY: tiny shared helpers. Every number on the proto pages comes from /api.
export async function api(path) {
  const r = await fetch(path);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
}
export const GLYPH = { Low: "▁", Medium: "▃", High: "▅", Critical: "▇" };
export const TIERVAR = (t) => `var(--tier-${t.toLowerCase()})`;
export const fmt1 = (n) => (Math.round(n * 10) / 10).toLocaleString("en-GB");
export const fmt0 = (n) => Math.round(n).toLocaleString("en-GB");
export const ward = (id) => id.replace(/_Ward_0?/, " · Ward ");
export const date = (iso) => new Date(iso + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
export const params = () => {
  const u = new URLSearchParams(location.search);
  return { ward: u.get("ward") || "DSCC_Ward_04", as_of: u.get("as_of") || "2025-07-14", h: +(u.get("h") || 3) };
};
export function tierHTML(t) {
  return `<span class="tier" style="color:${TIERVAR(t)}"><span class="g" aria-hidden="true">${GLYPH[t]}</span>${t}</span>`;
}
