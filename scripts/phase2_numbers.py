"""Print every Phase 2 number quoted in ARCHITECTURE.md, straight from the code.

    python -m scripts.phase2_numbers          # from the repo root; about a minute

Read-only: uses the committed model and data, writes nothing.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src.allocation import load_config, plan, totals
from src.intervention import input_floor, roi_ranking, simulate
from src.overflow import bed_capacity, overflow_plan
from src.predict import build_features, predict_all
from src.supplies import ONSITE_XLSX, SUPPLY_ITEMS, load_model, supplies_for

SCAN_FROM, SCAN_TO = "2024-06-03", "2025-11-24"
PEAK_WEEKS = ("2025-07-14", "2025-09-01")


def main() -> None:
    table, ratios = load_model()
    print("supplies: distinct case counts in data:", len(table), "max:", table.index.max(),
          "ratios:", {k: round(v, 3) for k, v in ratios.items()})
    data = pd.read_excel(ONSITE_XLSX, usecols=["dengue_cases", *SUPPLY_ITEMS])
    got = supplies_for(data["dengue_cases"].to_numpy())
    exact = int((got[SUPPLY_ITEMS].to_numpy() == data[SUPPLY_ITEMS].to_numpy()).all(axis=1).sum())
    print("supplies: rows reproduced exactly:", exact, "of", len(data))
    feat = build_features()
    rose = checked = 0                          # does lowering water-logging ever raise a forecast?
    biggest = 0.0
    for week in PEAK_WEEKS:
        for cut in (0.15, 0.30, 0.50):
            lower = feat.assign(water_logging_risk_index=feat["water_logging_risk_index"] * (1 - cut))
            for h in (2, 3, 4):
                b = predict_all(feat, week, h, with_drivers=False).set_index("ward_id")["predicted_cases"]
                a = predict_all(lower, week, h, with_drivers=False).set_index("ward_id")["predicted_cases"]
                d = a - b.reindex(a.index)
                rose += int((d > 1e-9).sum())
                checked += len(d)
                biggest = max(biggest, float(d.max()))
    print(f"water-logging cut 15/30/50% on {', '.join(PEAK_WEEKS)}, h=2,3,4: forecasts that rose "
          f"{rose} of {checked}, biggest rise {biggest:.2f} cases")
    print("water_logging distinct values per ward (max):",
          feat.groupby("ward_id")["water_logging_risk_index"].nunique().max())
    print("training-minimum water_logging:", input_floor(feat, "water_logging_risk_index"))
    cap = bed_capacity()
    print("bed capacity per ward: min", min(cap.values()), "max", max(cap.values()), "total", sum(cap.values()))

    a = plan(feat, "2025-07-14", 3, 50, crews=20)
    t = totals(a, 50, 20)
    print("allocation 2025-07-14 h3 50% 20 crews:",
          {k: (v["allocated"], v["required"]) for k, v in t["items"].items()},
          "critical", t["critical_wards"], "fully served", int((~a["critical"]).sum()), "crews", t["crews"]["assigned"])

    dates = sorted(d for d in feat["date"].unique() if pd.Timestamp(SCAN_FROM) <= d <= pd.Timestamp(SCAN_TO))
    print("overflow scan weeks:", len(dates), str(dates[0])[:10], "..", str(dates[-1])[:10])
    for pct in (100, 70, 50):
        n = sum(bool(overflow_plan(feat, d, 2, beds_available_pct=pct)["transfers"]) for d in dates)
        print(f"  {pct}%: weeks with a transfer {n}/{len(dates)}")
    r = overflow_plan(feat, "2025-09-15", 2, beds_available_pct=70)
    print("2025-09-15 h2 70%: moved", sum(x["patients"] for x in r["transfers"]),
          "stranded", sum(c["unroutable"] for c in r["cities"].values()))
    r = overflow_plan(feat, "2025-07-14", 2, beds_available_pct=50)
    w = list(r["wards"].values())
    print("2025-07-14 h2 50%: wards over", sum(x["beds_required"] > x["capacity"] for x in w),
          "wards with a free bed", sum(x["capacity"] > x["beds_required"] for x in w),
          "transfers", len(r["transfers"]),
          "city utilisation %", {k: round(v["utilisation_pct"]) for k, v in r["cities"].items()})

    t0 = time.time()
    rk = roi_ranking(feat, "2025-07-14")
    dt = time.time() - t0
    print("ROI 2025-07-14: top", rk[0]["ward_id"], rk[0]["roi"], "| 5th", rk[4]["roi"],
          "| last", rk[-1]["roi"], f"| {dt:.2f}s")
    print("  wards at the floor with 1 crew:", sum(x["floor_hit"] for x in rk))
    s = simulate(feat, "2025-07-14", {x["ward_id"]: 2 for x in rk[:5]})
    for h in s["horizons"]:
        print("  sim 10 crews h", h["horizon"], h["cases_before"], "->", h["cases_after"],
              "beds freed", h["beds_freed"], "saline saved", h["saline_saved"])
    one = simulate(feat, "2025-07-14", {rk[0]["ward_id"]: 2})["wards"][0]
    print("  same top ward, 2 crews: roi", one["roi"])
    q = roi_ranking(feat, "2025-02-03")
    print("off-season 2025-02-03 max cases prevented per ward:", max(x["cases_prevented"] for x in q))

    print("\nassumptions (config/operations.json):")
    for sec, entries in load_config().items():
        if sec.startswith("_"):
            continue
        for k, v in entries.items():
            print(f"  {sec}.{k} = {json.dumps(v['value'])} ({v['kind']})")


if __name__ == "__main__":
    main()
