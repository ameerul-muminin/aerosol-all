"""Build bilingual, offline dispatch notices from the operations plans."""

from datetime import date
from html import escape
from numbers import Integral, Real

import pandas as pd

from src import allocation, overflow
from src.features import HORIZONS, required_columns

TEMPLATES = {
    "en": {
        "title": "Dengue dispatch brief for the city commissioner",
        "ward": "{ward_id} · {city}",
        "forecast": "Week of {forecast_week}: {predicted_cases} cases (80% range {range_low}–{range_high}), {tier} risk",
        "short_forecast": "{forecast_week} {predicted_cases} cases ({range_low}–{range_high}, 80%) {tier}",
        "supply_heading": "Supplies (allocated/required)",
        "supply": "{label}: {allocated}/{required} · deficit {deficit}",
        "items": {"saline": "Saline bags", "kits": "Diagnostic kits", "beds": "Beds (stock plan)"},
        "crews": "Spraying crews: {crews_assigned}",
        "routing": "Hospital beds: {beds_required} needed, {capacity} available. Patients in {routed_in}, out {routed_out}, unroutable {unroutable}.",
        "to": "{patients} patients to {to_ward}",
        "from": "{patients} patients from {from_ward}",
        "alert": "CRITICAL DEFICIT",
        "ok": "No deficit",
        "sms_supply": "S {saline_allocated}/{saline_required} K {kits_allocated}/{kits_required} B {beds_allocated}/{beds_required}",
        "sms_routing": "crews {crews_assigned} in/out {routed_in}/{routed_out} unroutable {unroutable}",
        "intervention": "Spraying simulation (assumed effect, {crews} crews): {cases_before} → {cases_after} cases.",
        "tiers": {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"},
        "cities": {},
    },
    "bn": {
        "title": "সিটি কমিশনারের জন্য ডেঙ্গু কার্যক্রমের প্রতিবেদন",
        "ward": "{ward_id} · {city}",
        "forecast": "{forecast_week} সপ্তাহে {predicted_cases} জন (৮০% সীমা {range_low}–{range_high}), ঝুঁকি: {tier}",
        "short_forecast": "{forecast_week}: {predicted_cases} জন ({range_low}–{range_high}, ৮০%) · ঝুঁকি {tier}",
        "supply_heading": "সরঞ্জাম (বরাদ্দ/প্রয়োজন)",
        "supply": "{label}: {allocated}/{required} · ঘাটতি {deficit}",
        "items": {"saline": "স্যালাইনের ব্যাগ", "kits": "পরীক্ষার কিট", "beds": "শয্যা (মজুত পরিকল্পনা)"},
        "crews": "স্প্রে দল: {crews_assigned}",
        "routing": "হাসপাতালের শয্যা: প্রয়োজন {beds_required}, উপলব্ধ {capacity}। রোগী আসবে {routed_in}, যাবে {routed_out}, শয্যা পাবে না {unroutable} জন।",
        "to": "{patients} জন রোগী যাবেন {to_ward} ওয়ার্ডে",
        "from": "{patients} জন রোগী আসবেন {from_ward} ওয়ার্ড থেকে",
        "alert": "জরুরি ঘাটতি",
        "ok": "কোনো ঘাটতি নেই",
        "sms_supply": "স্যালাইন {saline_allocated}/{saline_required} কিট {kits_allocated}/{kits_required} শয্যা {beds_allocated}/{beds_required}",
        "sms_routing": "দল {crews_assigned} আসবে/যাবে {routed_in}/{routed_out} শয্যা পাবে না {unroutable} জন",
        "intervention": "স্প্রে কার্যক্রমের সিমুলেশন (অনুমিত প্রভাব, {crews} দল): রোগী {cases_before} → {cases_after} জন।",
        "tiers": {"Low": "কম", "Medium": "মাঝারি", "High": "বেশি", "Critical": "সংকটজনক"},
        "cities": {
            "Chattogram": "চট্টগ্রাম", "Dhaka North": "ঢাকা উত্তর", "Dhaka South": "ঢাকা দক্ষিণ",
            "Gazipur": "গাজীপুর", "Narayanganj": "নারায়ণগঞ্জ",
        },
    },
    "format": {
        "sms_join": " | ",
        "ellipsis": "…",
        "bold": "*{}*",
        "section": '<section class="{status}"><h2>{heading}</h2><p><strong>{alert}</strong></p>{details}</section>',
        "paragraph": "<p>{}</p>",
        "page": """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font: 11pt/1.4 system-ui, sans-serif; max-width: 48rem; margin: 2rem auto; padding: 0 1rem; }}
section {{ border-top: 1px solid #bbb; padding: .55rem 0; }}
h2 {{ font-size: 1.1rem; margin: 0 0 .3rem; }}
p {{ margin: .15rem 0; }}
.critical strong {{ color: #8c1717; }}
@page {{ margin: 15mm; }}
@media print {{
  body {{ max-width: none; margin: 0; padding: 0; }}
  section {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<h1>{title}</h1>
{sections}
</body>
</html>
""",
    },
}


def build_notices(feat_df, as_of_date, horizon, stock_pct, crews, beds_available_pct=100,
                  lang="en", *, intervention_results=None):
    """Return one notice dict per ward in allocation priority order.

    Supply columns are copied from allocation. The nested `overflow` entry
    records actual routing against available capacity. `critical` flags a
    stock deficit or unroutable patients. Optionally pass the result of
    intervention.simulate() for this as-of date as a separate scenario.
    """
    if lang not in ("en", "bn"):
        raise ValueError("Choose 'en' (English) or 'bn' (Bangla).")
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in HORIZONS:
        raise ValueError("Horizon must be 2, 3 or 4 weeks.")
    horizon = int(horizon)
    columns = {"date", "ward_id", "city", "population_density_sqkm", *required_columns()}
    if not isinstance(feat_df, pd.DataFrame) or feat_df.empty or not columns.issubset(feat_df.columns):
        raise ValueError("Provide ward forecast features from build_features().")
    try:
        as_of = pd.Timestamp(as_of_date)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Enter a valid Monday date, for example 2025-09-15.") from None
    if pd.isna(as_of) or as_of.tz is not None or as_of.weekday() != 0 or as_of != as_of.normalize():
        raise ValueError("Enter a valid Monday date, for example 2025-09-15.")

    cfg = allocation.load_config()
    stock_pct = allocation._check_stock_pct(stock_pct, cfg)
    if crews is not None and (not isinstance(crews, Real) or not float("-inf") < crews < float("inf")):
        raise ValueError("Number of spraying crews must be a whole number, 0 or more.")
    crews = allocation._check_crews(crews, cfg)
    rows = allocation.plan(feat_df, as_of, horizon, stock_pct, crews, config=cfg)
    routing = overflow.overflow_plan(feat_df, as_of, horizon, config=cfg,
                                     beds_available_pct=beds_available_pct)
    transfers = {ward_id: [] for ward_id in routing["wards"]}
    for transfer in routing["transfers"]:
        for ward_id in (transfer["from_ward"], transfer["to_ward"]):
            transfers[ward_id].append(transfer.copy())

    notices = rows.sort_values(["priority_rank", "ward_id"]).to_dict("records")
    scenarios = _intervention_by_ward(intervention_results, as_of.date(), horizon, transfers)
    for notice in notices:
        ward_id = notice["ward_id"]
        ward_routing = routing["wards"][ward_id]
        notice.update(lang=lang, forecast_week=str(notice["forecast_week"]),
                      overflow=ward_routing, transfers=transfers[ward_id],
                      unroutable=ward_routing["net_deficit"],
                      allocation_critical=notice["critical"],
                      critical=bool(notice["critical"] or ward_routing["net_deficit"]),
                      intervention=scenarios.get(ward_id))
    return notices


def _intervention_by_ward(results, as_of: date, horizon, ward_ids):
    if results is None:
        return {}
    try:
        if results["as_of"] != as_of.isoformat():
            raise ValueError("Intervention results must use the same as-of date as the notices.")
        scenarios = {}
        for ward in results["wards"]:
            ward_id = ward["ward_id"]
            if ward_id not in ward_ids or ward_id in scenarios:
                raise ValueError("Intervention results must reference each known ward at most once.")
            scenario = ward["by_horizon"][str(horizon)]
            scenarios[ward_id] = {
                "crews": int(ward["crews"]),
                "cases_before": float(scenario["cases_before"]),
                "cases_after": float(scenario["cases_after"]),
            }
    except (KeyError, TypeError, OverflowError):
        raise ValueError("Provide intervention.simulate() results for the selected horizon.") from None
    return scenarios


def _wording(notice):
    template = TEMPLATES[notice["lang"]]
    values = dict(notice, city=template["cities"].get(notice["city"], notice["city"]),
                  tier=template["tiers"][notice["risk_tier"]])
    return template, values


def _details(notice, template, values):
    yield template["forecast"].format(**values)
    yield template["supply_heading"]
    for item, label in template["items"].items():
        yield template["supply"].format(label=label, allocated=notice[f"{item}_allocated"],
                                        required=notice[f"{item}_required"], deficit=notice[f"{item}_deficit"])
    yield template["crews"].format(**values)
    yield template["routing"].format(**notice["overflow"], unroutable=notice["unroutable"])
    for transfer in notice["transfers"]:
        direction = "to" if transfer["from_ward"] == notice["ward_id"] else "from"
        yield template[direction].format(**transfer)
    if notice["intervention"] is not None:
        yield template["intervention"].format(**notice["intervention"])


def sms(notice):
    """Render plain text, preserving the critical deficit alert when shortened."""
    template, values = _wording(notice)
    parts = [
        template["alert"] if notice["critical"] else template["ok"],
        template["ward"].format(**values),
        template["short_forecast"].format(**values),
        template["sms_supply"].format(**values),
        template["sms_routing"].format(**values, routed_in=notice["overflow"]["routed_in"],
                                       routed_out=notice["overflow"]["routed_out"]),
    ]
    if notice["intervention"] is not None:
        parts.append(template["intervention"].format(**notice["intervention"]))
    fmt = TEMPLATES["format"]
    message = fmt["sms_join"].join(parts)
    if notice["lang"] == "en" and len(message) > 160:
        suffix = fmt["ellipsis"]
        return message[:160 - len(suffix)].rstrip() + suffix
    return message


def whatsapp(notice):
    """Render a multiline WhatsApp notice with bold headings."""
    template, values = _wording(notice)
    bold = TEMPLATES["format"]["bold"]
    heading = bold.format(template["ward"].format(**values))
    alert = bold.format(template["alert"] if notice["critical"] else template["ok"])
    return "\n".join((heading, alert, *_details(notice, template, values)))


def brief_html(notices):
    """Render a single printable brief with critical wards first."""
    lang = notices[0]["lang"] if notices else "en"
    if any(notice["lang"] != lang for notice in notices):
        raise ValueError("Use one language per printable brief.")
    fmt = TEMPLATES["format"]
    sections = []
    for notice in sorted(notices, key=lambda row: (not row["critical"], row["priority_rank"], row["ward_id"])):
        template, values = _wording(notice)
        sections.append(fmt["section"].format(
            status="critical" if notice["critical"] else "routine",
            heading=escape(template["ward"].format(**values)),
            alert=escape(template["alert"] if notice["critical"] else template["ok"]),
            details="\n".join(fmt["paragraph"].format(escape(line)) for line in _details(notice, template, values)),
        ))
    return fmt["page"].format(lang=lang, title=escape(TEMPLATES[lang]["title"]),
                              sections="\n".join(sections))
