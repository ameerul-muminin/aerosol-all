---
name: build-ui
description: How to build or change anything in web/ (dashboard or marketing page). Night-direction tokens, plain HTML/CSS/vanilla JS, data only from /api, Emil Kowalski motion rules. Use for every frontend task.
---

# Build UI (web/)

## Before writing code
1. Read `web/assets/css/tokens.css`. Use variables only; never hard-code a colour, radius or duration.
2. Look at the matching prototype artboard (Night A marketing, Night B ward outlook, Night C all wards).
   Ask the human for the canvas link if you need it.
3. Search `web/assets/js/` for an existing helper before writing a new one.

## Stack rules
- Plain HTML + CSS + vanilla JS ES modules. No npm, no bundler, no framework, no CDN at runtime.
- Fonts and images live in `web/assets/`. Outfit woff2 files in `web/assets/fonts/`.
- Data comes ONLY from `/api/*` via `web/assets/js/api.js`. Never type a number into HTML.
  Placeholders while loading are skeletons, not fake values.
- State in the URL: `/app/?ward=CCC_Ward_01&as_of=2025-07-14&h=3#outlook`.

## Content rules (judges score these)
- Risk tier = colour + glyph + word: Low ▁, Medium ▃, High ▅, Critical ▇.
- The "Why this forecast" driver breakdown must be visible on the ward view (UI score halves without it).
  Show "Average week X cases → this ward Y" from `driver_reference_cases`.
- Bonus items: ward × horizon heatmap (All wards), likely range on every forecast, MAE/RMSE accuracy view.
- Marketing copy may quote only: 150,000+ annual hospitalizations, 7–14 day breeding, 2–4 week lag
  (credit "EDU AI Hackathon 2026 brief"). Every other number is fetched.

## Motion (Emil Kowalski)
- Animate `transform` and `opacity` only. Bars grow with `transform: scaleY()` from `transform-origin: bottom`.
- Ease-out `var(--ease-out)`; press 160 ms, moves 240 ms, bars 320 ms. Nothing slower than 400 ms.
- Buttons: `:active { transform: scale(.97) }`. Segmented control: one sliding thumb (`translateX`).
- No entrance animations on every section; one subtle rain layer on the background at most.
- Keyboard-triggered changes don't animate layout. Respect `prefers-reduced-motion` (tokens set durations to 0).

## Accessibility
Real `<button>`, `<a href>`, `<label>` + `<select>`; visible focus ring (`--accent`); text contrast ≥ 4.5:1
(use `--text`, `--text-2`, `--text-3` only); touch targets ≥ 44 px; charts get `role="img"` + `aria-label`.

## Finish
Run the `verify` skill (steps 1, 2, 4). Commit small: `feat(web): <what>`.
