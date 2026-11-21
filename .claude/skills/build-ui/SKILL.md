---
name: build-ui
description: How to build or change anything in web/ (dashboard or marketing page). Warm paper ("Jute") tokens, plain HTML/CSS/vanilla JS, EN/বাংলা i18n, data only from /api, Emil Kowalski motion rules. Use for every frontend task.
---

# Build UI (web/)

## Before writing code
1. Read `web/assets/css/tokens.css`. Use variables only; never hard-code a colour, radius or duration.
2. Look at the locked prototypes: `web/proto/hero-a.html` (marketing) and `web/proto/outlook-b.html` (ward outlook).
3. Search `web/assets/js/` for an existing helper before writing a new one.

## Design (locked 22 Nov): warm paper, "Jute"
- **Feel:** calm field tool, not a sci-fi dashboard. Warm off-white paper, warm charcoal ink, ONE muted mustard
  accent (`--accent`) for the primary button, the breeding window and the likely range. Nothing neon, no gradients.
- **Type:** Anton (uppercase, line-height ≈ 0.98) only for page titles and big numbers. Everything else Hind
  Siliguri 400/500/600, sentence case. Card titles are 15 px / 600, not display type. Bangla (`lang="bn"`):
  Anton has no Bangla glyphs, so `.display` switches to Hind Siliguri 700, no uppercase, line-height 1.15.
- **Surfaces:** cards = `--paper` + `--shadow-sm` (hairline + 1 px drop). No coloured stripes, no glows, no
  decorative circles or icon tiles, no glassmorphism. Dividers are 1 px `--line`.
- **Layout:** dashboard = light sidebar (`--sunk`, 248 px: brand, view tabs, Ward + As of, EN/বাংলা) and a
  12-column card grid. Under 900 px the sidebar folds into a header and cards stack.
- **Tiers:** colour + 4-bar glyph (`.tg`, filled up to the level) + word. Tier names come from i18n.
- **Copy:** short. A card title, the number, one line of context. No helper paragraphs, no marketing tone in the app.
  Every string goes through `t()` (`web/assets/i18n/en.json` + `bn.json`); driver labels and tier names are
  translated in the frontend (`driver.*`, `tier.*` keys), never shown raw from the API except as a fallback.
- **Illustration:** marketing hero = `web/assets/images/hero.webp` (+ `hero-900.webp`). No other photos.

## Stack rules
- Plain HTML + CSS + vanilla JS ES modules. No npm, no bundler, no framework, no CDN at runtime.
- Fonts and images live in `web/assets/`: Anton + Hind Siliguri woff2 (OFL) in `web/assets/fonts/`.
- Data comes ONLY from `/api/*` via `web/assets/js/api.js`. Never type a number into HTML.
  Placeholders while loading are skeletons, not fake values.
- State in the URL: `/app/?ward=CCC_Ward_01&as_of=2025-07-14&h=3#outlook`.

## Content rules (judges score these)
- Risk tier = colour + 4-bar glyph + word (`tierHTML()` in `web/assets/js/format.js`).
- The "Why this forecast" driver breakdown must be visible on the ward view (UI score halves without it).
  Show "Average week X cases → this ward Y" from `driver_reference_cases`.
- Bonus items: ward × horizon heatmap (All wards), likely range on every forecast, MAE/RMSE accuracy view.
- Marketing copy may quote only: 150,000+ annual hospitalizations, 7–14 day breeding, 2–4 week lag
  (credit "EDU AI Hackathon 2026 brief"). Every other number is fetched.

## Motion (Emil Kowalski)
- Animate `transform` and `opacity` only. Bars grow with `transform: scaleY()` from `transform-origin: bottom`.
- Ease-out `var(--ease-out)`; press 160 ms, moves 240 ms, bars 320 ms. Nothing slower than 400 ms.
- Buttons: `:active { transform: scale(.97) }`. Segmented control: one sliding thumb (`translateX`).
- No entrance animations on every section. Bars grow on the first paint of a view only, not on every control change.
- Keyboard-triggered changes don't animate layout. Respect `prefers-reduced-motion` (tokens set durations to 0).

## Accessibility
Real `<button>`, `<a href>`, `<label>` + `<select>`; visible focus ring (`--accent`); text contrast ≥ 4.5:1
(use `--ink`, `--ink-2`, `--ink-3` only); touch targets ≥ 44 px; charts get `role="img"` + `aria-label`.

## Finish
Run the `verify` skill (steps 1, 2, 4). Commit small: `feat(web): <what>`.
