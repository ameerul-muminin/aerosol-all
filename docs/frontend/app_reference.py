"""Ward dengue outlook: Streamlit dashboard (reference implementation for Person C).

Run:  streamlit run app.py
Every number on screen comes from the trained model (src/predict.py) or reports/metrics.json.
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.predict import build_features, load_bundle, predict_all

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "reports" / "metrics.json"
HORIZONS = (2, 3, 4)
MIN_HISTORY_WEEKS = 8

# ---- design tokens -------------------------------------------------------------
INK = "#14324A"          # monsoon navy: headings, text
MUTED = "#5B6B78"        # secondary text
RAIN = "#2A78D6"         # rain bars, primary widget colour
RAIN_LIGHT = "#9EC5F4"   # rain outside the breeding window
RAIN_DEEP = "#104281"    # rain inside the breeding window
RAISES = "#E34948"       # driver raises risk
LOWERS = "#2A78D6"       # driver lowers risk
TIERS = {                # status palette: colour + glyph + label, never colour alone
    "Low":      {"color": "#0CA30C", "glyph": "▁", "tint": "#E7F6E7"},
    "Medium":   {"color": "#FAB219", "glyph": "▃", "tint": "#FEF4DD"},
    "High":     {"color": "#EC835A", "glyph": "▅", "tint": "#FCEBE3"},
    "Critical": {"color": "#D03B3B", "glyph": "▇", "tint": "#F9E2E2"},
}
TIER_ORDER = list(TIERS)
SEQ_BLUE = ["#CDE2FB", "#9EC5F4", "#6DA7EC", "#3987E5", "#256ABF", "#184F95", "#0D366B"]

st.set_page_config(page_title="Ward dengue outlook", page_icon="🦟", layout="wide")

st.markdown(f"""
<style>
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.lede {{ color: {MUTED}; font-size: 1.05rem; margin-top: -0.6rem; }}
.tier {{ display: flex; align-items: center; gap: 0.75rem; padding: 0.9rem 1.1rem;
         border-left: 6px solid var(--tc); background: var(--tt); border-radius: 6px; }}
.tier .glyph {{ font-size: 2.2rem; line-height: 1; color: var(--tc); }}
.tier .name {{ font-size: 1.6rem; font-weight: 700; color: {INK}; }}
.tier .what {{ color: {MUTED}; font-size: 0.9rem; }}
.chip {{ display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px;
         background: var(--tt); border-left: 4px solid var(--tc); color: {INK}; }}
</style>
""", unsafe_allow_html=True)


# ---- cached data + model (loaded once; widget clicks never recompute these) ----------
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_bundle()


@st.cache_data(show_spinner="Preparing weather lag features…")
def get_features() -> pd.DataFrame:
    return build_features()


@st.cache_data(show_spinner=False)
def get_ranking(as_of: str, horizon: int) -> pd.DataFrame:
    """All 60 wards for one week and horizon, one vectorised model call, cached per choice."""
    return predict_all(get_features(), as_of, horizon)


@st.cache_data
def get_metrics() -> dict:
    return json.loads(METRICS.read_text())


@st.cache_data
def get_test_predictions(horizon: int) -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / f"test_predictions_h{horizon}.csv", parse_dates=["target_date"])


def tier_badge(tier: str) -> str:
    t = TIERS[tier]
    return (f'<div class="tier" style="--tc:{t["color"]};--tt:{t["tint"]}">'
            f'<span class="glyph" aria-hidden="true">{t["glyph"]}</span>'
            f'<div><div class="name">{tier}</div><div class="what">risk tier</div></div></div>')


get_model()
feat = get_features()
first_date, last_date = feat["date"].min(), feat["date"].max()
valid_dates = [pd.Timestamp(d) for d in sorted(feat["date"].unique())
               if pd.Timestamp(d) >= first_date + pd.Timedelta(weeks=MIN_HISTORY_WEEKS)]

# ---- sidebar inputs ------------------------------------------------------------------
with st.sidebar:
    st.header("Choose a ward")
    cities = sorted(feat["city"].unique())
    city = st.selectbox("City corporation", cities, index=cities.index("Chattogram") if "Chattogram" in cities else 0)
    wards = sorted(feat.loc[feat["city"] == city, "ward_id"].unique())
    ward = st.selectbox("Ward", wards)
    default = pd.Timestamp("2025-07-14") if pd.Timestamp("2025-07-14") in valid_dates else valid_dates[-1]
    as_of = st.select_slider("Forecast from the week of", options=valid_dates, value=default,
                             format_func=lambda d: d.strftime("%d %b %Y"))
    horizon = st.radio("How far ahead", HORIZONS, index=1, horizontal=True, format_func=lambda h: f"{h} weeks")
    st.caption("Uses weather and case data up to the chosen week only.")
    with st.expander("How to read this"):
        st.markdown(
            "- **Risk tier** groups the forecast into Low, Medium, High or Critical.\n"
            "- **Likely range** is an 80% interval: in testing, the real count fell inside it about 8 times in 10.\n"
            "- **Why** shows what pushed this forecast up or down compared with typical conditions.\n"
            "- Rain takes 2–4 weeks to turn into mosquitoes, so the forecast reads the last weeks of rain.")

as_of = pd.Timestamp(as_of)
as_of_key = as_of.strftime("%Y-%m-%d")
forecast_week = as_of + pd.Timedelta(weeks=horizon)

st.title("Ward dengue outlook")
st.markdown(f'<p class="lede">Forecast for the week of {forecast_week:%d %b %Y}, '
            f'{horizon} weeks after {as_of:%d %b %Y}</p>', unsafe_allow_html=True)

try:
    ranking = get_ranking(as_of_key, horizon)
except ValueError as err:          # friendly message, never a traceback
    st.warning(str(err))
    st.stop()

tab_ward, tab_all, tab_acc = st.tabs(["Ward outlook", "All wards", "Model accuracy"])

# ---- tab 1: one ward ------------------------------------------------------------------
with tab_ward:
    row = ranking.loc[ranking["ward_id"] == ward]
    if row.empty:
        st.warning("No forecast for this ward on that week. Pick a later week.")
        st.stop()
    r = row.iloc[0]
    st.subheader(f"{ward}, {city}")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.markdown(tier_badge(r["risk_tier"]), unsafe_allow_html=True)
    c2.metric("Predicted cases", f"{r['predicted_cases']:.1f}")
    c3.metric("Likely range (80%)", f"{r['range_low']:.0f} – {r['range_high']:.0f}")

    # Signature chart: rain → risk timeline
    st.markdown("#### From rain to risk")
    hist = feat[(feat["ward_id"] == ward) & (feat["date"] <= as_of)].tail(MIN_HISTORY_WEEKS + 1).copy()
    win_lo, win_hi = forecast_week - pd.Timedelta(weeks=4), forecast_week - pd.Timedelta(weeks=2)
    hist["window"] = hist["date"].between(win_lo, win_hi).map(
        {True: "Breeding window (2–4 weeks before)", False: "Other weeks"})
    rain = alt.Chart(hist).mark_bar(cornerRadiusEnd=4, size=18).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
        y=alt.Y("rainfall_mm:Q", title="Rainfall (mm)", axis=alt.Axis(gridColor="#E6EBEF")),
        color=alt.Color("window:N", title=None,
                        scale=alt.Scale(domain=["Breeding window (2–4 weeks before)", "Other weeks"],
                                        range=[RAIN_DEEP, RAIN_LIGHT]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("date:T", title="Week", format="%d %b %Y"),
                 alt.Tooltip("rainfall_mm:Q", title="Rain (mm)", format=".1f"),
                 alt.Tooltip("dengue_cases:Q", title="Cases that week")],
    )
    target = pd.DataFrame({"date": [forecast_week], "label": [f"Forecast week: {r['predicted_cases']:.1f} cases"]})
    rule = alt.Chart(target).mark_rule(color=TIERS[r["risk_tier"]]["color"], strokeWidth=2, strokeDash=[4, 3]).encode(x="date:T")
    text = alt.Chart(target).mark_text(align="right", dx=-6, dy=-8, color=INK).encode(
        x="date:T", y=alt.value(12), text="label:N")
    st.altair_chart((rain + rule + text).properties(height=260), use_container_width=True)
    st.caption("Dark bars are the weeks whose rain matters most for the forecast week. "
               "Rain after the chosen week is not known yet, so it is not used.")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Why this forecast")
        drv = pd.DataFrame(r["top_drivers"], columns=["factor", "effect"])
        drv["direction"] = drv["effect"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
        bars = alt.Chart(drv).mark_bar(cornerRadiusEnd=4, size=16).encode(
            x=alt.X("effect:Q", title="Change in predicted cases vs typical conditions",
                    axis=alt.Axis(gridColor="#E6EBEF")),
            y=alt.Y("factor:N", sort=alt.EncodingSortField("effect", op="max", order="descending"),
                    title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("direction:N", title=None,
                            scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RAISES, LOWERS]),
                            legend=alt.Legend(orient="top")),
            tooltip=["factor", "direction", alt.Tooltip("effect:Q", format="+.2f", title="Cases")],
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
        st.altair_chart((bars + zero).properties(height=210), use_container_width=True)
    with right:
        st.markdown("#### What the model saw")
        recent = hist.tail(5)[["date", "rainfall_mm", "avg_temp_c", "relative_humidity_pct", "dengue_cases"]]
        recent = recent.rename(columns={"date": "Week", "rainfall_mm": "Rain (mm)", "avg_temp_c": "Temp (°C)",
                                        "relative_humidity_pct": "Humidity (%)", "dengue_cases": "Cases"})
        recent["Week"] = recent["Week"].dt.strftime("%d %b")
        st.dataframe(recent, hide_index=True, use_container_width=True)
        w = feat.loc[feat["ward_id"] == ward].iloc[0]
        st.caption(f"Ward profile: water-logging risk {w['water_logging_risk_index']:.2f}, "
                   f"sanitation score {w['sanitation_score']:.2f}, "
                   f"{w['population_density_sqkm']:,} people per km².")

# ---- tab 2: all wards ----------------------------------------------------------------
with tab_all:
    st.markdown("#### Where to act first")
    pick = st.multiselect("Cities", sorted(ranking["city"].unique()), default=sorted(ranking["city"].unique()))
    view = ranking[ranking["city"].isin(pick)].copy()

    counts = view["risk_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    cols = st.columns(4)
    for col, tier in zip(cols, TIER_ORDER):
        t = TIERS[tier]
        col.markdown(f'<span class="chip" style="--tc:{t["color"]};--tt:{t["tint"]}">'
                     f'{t["glyph"]} {tier}: <b>{counts[tier]}</b> wards</span>', unsafe_allow_html=True)

    # Ward × horizon heatmap: predicted cases 2, 3 and 4 weeks out (no coordinates, so no map)
    heat = pd.concat([get_ranking(as_of_key, h).assign(ahead=f"+{h} weeks") for h in HORIZONS])
    heat = heat[heat["city"].isin(pick)]
    order = view["ward_id"].tolist()
    hm = alt.Chart(heat).mark_rect(stroke="#FCFCFB", strokeWidth=2).encode(
        x=alt.X("ahead:O", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("ward_id:N", sort=order, title=None),
        color=alt.Color("predicted_cases:Q", title="Predicted cases", scale=alt.Scale(range=SEQ_BLUE)),
        tooltip=["ward_id", "city", "ahead", alt.Tooltip("predicted_cases:Q", format=".1f"), "risk_tier"],
    ).properties(height=max(14 * len(order), 120))

    a, b = st.columns([1, 1.6])
    with a:
        st.altair_chart(hm, use_container_width=True)
    with b:
        table = view[["ward_id", "city", "risk_tier", "predicted_cases", "range_low", "range_high"]].copy()
        table["risk_tier"] = table["risk_tier"].map(lambda t: f"{TIERS[t]['glyph']} {t}")
        table.columns = ["Ward", "City", "Risk tier", "Predicted cases", "Low", "High"]
        st.dataframe(table, hide_index=True, use_container_width=True, height=min(36 * (len(table) + 1), 640))
        st.download_button("Download ranking (CSV)", table.to_csv(index=False),
                           file_name=f"ward_ranking_{as_of_key}_h{horizon}.csv", mime="text/csv")

# ---- tab 3: model accuracy (all numbers from reports/) ----------------------------------
with tab_acc:
    m = get_metrics()
    st.markdown("#### How accurate is it?")
    st.caption(f"Test period {m['split']['test']}: weeks the model never saw while training or tuning.")
    rows = []
    for h in HORIZONS:
        t = m["horizons"][str(h)]["test"]
        rows.append({"Weeks ahead": h, "MAE (model)": t["MAE"], "RMSE (model)": t["RMSE"],
                     "MAE (seasonal baseline)": t["baseline_seasonal_naive"]["MAE"],
                     "80% range coverage": t["interval80_coverage"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)

    tp = get_test_predictions(horizon)
    by_city = (tp.assign(err=(tp["pred"] - tp["actual"]).abs(), sq=(tp["pred"] - tp["actual"]) ** 2)
                 .groupby("city").agg(MAE=("err", "mean"), RMSE=("sq", lambda s: s.mean() ** 0.5)).round(3))
    st.markdown(f"**Error by city, {horizon} weeks ahead**")
    st.dataframe(by_city, use_container_width=True)

    one = tp[tp["ward_id"] == ward]
    if not one.empty:
        st.markdown(f"**Predicted vs actual for {ward}, {horizon} weeks ahead (test period)**")
        band = alt.Chart(one).mark_area(color=RAIN_LIGHT, opacity=0.5).encode(
            x=alt.X("target_date:T", title=None), y=alt.Y("lo:Q", title="Cases"), y2="hi:Q")
        line_p = alt.Chart(one).mark_line(color=RAIN_DEEP, strokeWidth=2).encode(x="target_date:T", y="pred:Q")
        line_a = alt.Chart(one).mark_line(color=INK, strokeWidth=2, strokeDash=[3, 2]).encode(
            x="target_date:T", y="actual:Q",
            tooltip=[alt.Tooltip("target_date:T", title="Week"), "actual", alt.Tooltip("pred:Q", format=".1f")])
        st.altair_chart((band + line_p + line_a).properties(height=260), use_container_width=True)
        st.caption("Solid line: forecast. Dashed line: reported cases. Shaded: 80% likely range.")

st.divider()
st.caption("Decision support built on the hackathon dataset. It is not a medical diagnosis. "
           "Forecasts use only data up to the chosen week.")
