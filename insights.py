"""
Rule-based checks and Q&A over the filtered dashboard data.

find_issues(ctx)         runs a set of checks and returns ranked findings, each
                         with the evidence behind it and a suggested action.
answer_question(q, ctx)  matches a typed question to a topic and computes the
                         answer from the dataframes, so every number shown comes
                         from the data on screen.

Nothing here calls an external service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONTEXT
# --------------------------------------------------------------------------


@dataclass
class QueryContext:
    """Everything the checks can look at: the filtered view only."""
    price: pd.DataFrame          # filtered price+msp rows (has 'ym')
    arrivals: pd.DataFrame       # filtered arrival rows (has 'ym')
    transport: pd.DataFrame      # filtered transport rows
    weather: pd.DataFrame        # weather rows inside the date window
    valued: pd.DataFrame         # arrivals joined to price, with 'market_value'
    risk: pd.DataFrame           # per-mandi risk table (mandi_id, mandi_name, score, ...)
    master: pd.DataFrame
    metrics: dict = field(default_factory=dict)   # revenue, below-MSP rate, transit, etc.


@dataclass
class Finding:
    severity: str        # 'high' | 'medium' | 'low' | 'info'
    title: str
    finding: str
    evidence: str
    action: str
    rank: float = 0.0    # used for ordering; higher = more urgent


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _safe(fn, default=None):
    """Run a check; one failing check should not break the rest."""
    try:
        return fn()
    except Exception:
        return default


def _complete_months(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Monthly totals with a trailing PARTIAL month removed.

    Source data often ends mid-month (e.g. the 8th). Left in, that stub month
    looks like a collapse in volume and flips any trend calculation negative, so
    we would report a contraction that is really just a truncated month.
    """
    if df.empty or "ym" not in df.columns:
        return pd.Series(dtype=float)
    monthly = df.groupby("ym")[value_col].sum().sort_index()
    last_date = df["clean_date"].max()
    if pd.notna(last_date) and last_date.day < 25 and len(monthly) > 1:
        monthly = monthly.iloc[:-1]
    return monthly


# --------------------------------------------------------------------------
# 1. CHECKS ON THE SELECTION
# --------------------------------------------------------------------------
def find_issues(ctx: QueryContext) -> list[Finding]:
    out: list[Finding] = []

    # ---- check 1: MSP breach hotspot --------------------------------------
    def msp_hotspot():
        p = ctx.price.dropna(subset=["msp"]).copy()
        if p.empty:
            return None
        p["below"] = p["modal_price"] < p["msp"]
        net = p["below"].mean()
        by_mandi = p.groupby("mandi_id")["below"].agg(["mean", "count"])
        by_mandi = by_mandi[by_mandi["count"] >= 15]
        if by_mandi.empty:
            return None
        worst_id = by_mandi["mean"].idxmax()
        worst = by_mandi.loc[worst_id]
        if worst["mean"] <= net * 1.15:
            return None
        name = _mandi_name(ctx, worst_id)
        gap = (worst["mean"] - net) * 100
        return Finding(
            severity="high" if worst["mean"] > 0.5 else "medium",
            title="MSP breach hotspot",
            finding=(
                f"{name} clears below MSP far more often than the rest of the selected "
                f"network. Farmers selling there are the most exposed to guaranteed-price shortfall."
            ),
            evidence=(
                f"{worst['mean']*100:.1f}% of its {int(worst['count'])} recorded sales fell below MSP, "
                f"versus a network average of {net*100:.1f}%, a gap of {gap:.1f} percentage points."
            ),
            action=(
                f"Prioritise {name} for procurement-desk intervention: verify MSP enforcement, "
                f"check whether arrivals are arriving in gluts, and compare its buyer mix to lower-risk mandis."
            ),
            rank=90 + gap,
        )

    # ---- check 2: crop most exposed to MSP shortfall -----------------------
    def crop_exposure():
        p = ctx.price.dropna(subset=["msp"]).copy()
        if p.empty:
            return None
        p["below"] = p["modal_price"] < p["msp"]
        by_crop = p.groupby("crop_name")["below"].agg(["mean", "count"])
        by_crop = by_crop[by_crop["count"] >= 20]
        if by_crop.empty:
            return None
        crop = by_crop["mean"].idxmax()
        rate = by_crop.loc[crop, "mean"]
        if rate < 0.2:
            return None
        shortfall = p[(p["crop_name"] == crop) & p["below"]]
        avg_gap = (shortfall["msp"] - shortfall["modal_price"]).mean()
        return Finding(
            severity="high" if rate > 0.45 else "medium",
            title=f"{crop} is the most MSP-exposed crop",
            finding=(
                f"Across the current selection, {crop} sales fall short of MSP more often than any "
                f"other crop, and the shortfall is material rather than marginal."
            ),
            evidence=(
                f"{rate*100:.1f}% of {crop} sales priced below MSP "
                f"({int(by_crop.loc[crop,'count'])} records); when below, the average gap is "
                f"₹{avg_gap:,.0f}/quintal."
            ),
            action=(
                f"Model the cost of topping {crop} up to MSP; the per-quintal gap times the "
                f"below-MSP volume is the direct farmer-income loss this dashboard is tracking."
            ),
            rank=80 + rate * 20,
        )

    # ---- check 3: revenue momentum ----------------------------------------
    def revenue_trend():
        v = ctx.valued.dropna(subset=["market_value"])
        if v.empty or "ym" not in v.columns:
            return None
        monthly = _complete_months(v, "market_value")
        if len(monthly) < 3:
            return None
        x = np.arange(len(monthly))
        slope = np.polyfit(x, monthly.values, 1)[0]
        pct_per_month = 100 * slope / max(monthly.mean(), 1)
        if abs(pct_per_month) < 2:
            return Finding(
                severity="info", title="Market value is broadly flat",
                finding="Estimated market value shows no meaningful directional trend over the window.",
                evidence=f"Trend of {pct_per_month:+.1f}% per month across {len(monthly)} months.",
                action="Treat month-to-month moves as noise; compare against the same months last season instead.",
                rank=20,
            )
        direction = "growing" if pct_per_month > 0 else "contracting"
        return Finding(
            severity="low" if pct_per_month > 0 else "medium",
            title=f"Market value is {direction}",
            finding=(
                f"Estimated market value is {direction} month over month across the selected window."
            ),
            evidence=(
                f"Linear trend of {pct_per_month:+.1f}% per month; latest month "
                f"₹{monthly.iloc[-1]/1e7:,.2f} Cr vs first month ₹{monthly.iloc[0]/1e7:,.2f} Cr."
            ),
            action=(
                "Confirm whether the move is price-led or volume-led before acting. The Price and "
                "Arrivals tabs separate the two."
                if pct_per_month < 0 else
                "Check that logistics capacity scales with the rising volume; transit delay is the usual first constraint."
            ),
            rank=55 + abs(pct_per_month),
        )

    # ---- check 4: logistics bottleneck ------------------------------------
    def logistics_bottleneck():
        t = ctx.transport.dropna(subset=["clean_transit_hours"])
        if t.empty:
            return None
        net = t["clean_transit_hours"].mean()
        wh = t.groupby("destination_warehouse")["clean_transit_hours"].agg(["mean", "count"])
        wh = wh[wh["count"] >= 20]
        if wh.empty:
            return None
        worst = wh["mean"].idxmax()
        val = wh.loc[worst, "mean"]
        if val <= net * 1.15:
            return None
        return Finding(
            severity="medium" if val < net * 1.5 else "high",
            title="Logistics bottleneck at a destination warehouse",
            finding=(
                f"Consignments routed to {worst} sit in transit materially longer than the network "
                f"norm, which raises spoilage exposure for perishable lots."
            ),
            evidence=(
                f"Average transit to {worst} is {val:.1f} hrs across {int(wh.loc[worst,'count'])} trips, "
                f"versus {net:.1f} hrs network-wide ({(val/net-1)*100:+.0f}%)."
            ),
            action=(
                f"Audit the routes feeding {worst}. The Logistics tab's distance-vs-time scatter "
                f"separates genuinely long hauls from routes that are slow for their distance."
            ),
            rank=70 + (val / max(net, 1) - 1) * 40,
        )

    # ---- check 5: slow-for-distance routes --------------------------------
    def inefficient_routes():
        t = ctx.transport.dropna(subset=["clean_transit_hours", "distance_km"])
        t = t[t["distance_km"] > 0]
        if len(t) < 30:
            return None
        t = t.copy()
        t["hrs_per_100km"] = 100 * t["clean_transit_hours"] / t["distance_km"]
        med = t["hrs_per_100km"].median()
        bad = t[t["hrs_per_100km"] > med * 2]
        if bad.empty:
            return None
        share = len(bad) / len(t)
        if share < 0.05:
            return None
        top_mandi = bad["mandi_id"].value_counts().idxmax()
        return Finding(
            severity="medium" if share > 0.12 else "low",
            title="A cluster of routes is slow for its distance",
            finding=(
                "Some trips take far longer per kilometre than the network median, which points to "
                "route or handling problems rather than simple geography."
            ),
            evidence=(
                f"{share*100:.1f}% of trips ({len(bad)} of {len(t)}) run at more than twice the median "
                f"{med:.1f} hrs/100km. The most frequent origin is {_mandi_name(ctx, top_mandi)}."
            ),
            action=(
                "Pull the worst offenders by hrs/100km and check loading delays, intermediate stops, "
                "and vehicle type before assuming the road distance is the cause."
            ),
            rank=60 + share * 100,
        )

    # ---- check 6: volume concentration risk --------------------------------
    def concentration():
        a = ctx.arrivals
        if a.empty:
            return None
        by_mandi = a.groupby("mandi_id")["arrival_quantity_qtl"].sum().sort_values(ascending=False)
        if len(by_mandi) < 5:
            return None
        top3_share = by_mandi.head(3).sum() / by_mandi.sum()
        if top3_share < 0.35:
            return None
        names = ", ".join(_mandi_name(ctx, m) for m in by_mandi.head(3).index)
        return Finding(
            severity="medium" if top3_share > 0.5 else "low",
            title="Arrival volume is concentrated in a few mandis",
            finding=(
                "A small number of mandis absorb a disproportionate share of total arrivals, so a "
                "disruption at any one of them propagates across the network."
            ),
            evidence=f"The top 3 mandis handle {top3_share*100:.1f}% of all arrivals in scope: {names}.",
            action=(
                "Stress-test the network: if the largest of these went offline for a week, confirm "
                "neighbouring mandis have the throughput to absorb the diverted volume."
            ),
            rank=50 + top3_share * 40,
        )

    # ---- check 7: price volatility -----------------------------------------
    def volatility():
        p = ctx.price
        if p.empty:
            return None
        v = p.groupby("mandi_id")["modal_price"].agg(["mean", "std", "count"])
        v = v[v["count"] >= 20]
        if v.empty:
            return None
        v["cv"] = v["std"] / v["mean"]
        worst_id = v["cv"].idxmax()
        cv = v.loc[worst_id, "cv"]
        if cv < 0.25:
            return None
        return Finding(
            severity="medium" if cv > 0.4 else "low",
            title="Unstable pricing at one mandi",
            finding=(
                f"{_mandi_name(ctx, worst_id)} shows the widest price dispersion in the selection. "
                f"Volatility makes it hard for farmers to time their sale, which itself destroys value."
            ),
            evidence=(
                f"Coefficient of variation of {cv:.2f} on modal price "
                f"(mean ₹{v.loc[worst_id,'mean']:,.0f}, sd ₹{v.loc[worst_id,'std']:,.0f})."
            ),
            action=(
                "Check whether the spread is driven by crop mix rather than genuine instability. "
                "Filter to a single crop and re-read this figure."
            ),
            rank=45 + cv * 30,
        )

    # ---- check 8: weather pressure -----------------------------------------
    def weather_pressure():
        w = ctx.weather
        if w.empty or "temperature_celsius" not in w.columns:
            return None
        hot = (w["temperature_celsius"] > 35).mean()
        if hot < 0.15:
            return None
        return Finding(
            severity="medium" if hot > 0.3 else "low",
            title="Heat exposure across the window",
            finding=(
                "A meaningful share of sensor readings sit above 35°C, which raises post-harvest "
                "loss risk for produce waiting in open mandi yards."
            ),
            evidence=f"{hot*100:.1f}% of readings exceeded 35°C in the selected date range.",
            action=(
                "Cross-reference the hottest months against transit times, since long transit in high heat "
                "is where spoilage risk compounds."
            ),
            rank=40 + hot * 30,
        )

    # ---- check 9: thin data warning ----------------------------------------
    def coverage_gap():
        a, p = ctx.arrivals, ctx.price
        if a.empty:
            return None
        mandis_with_arrivals = set(a["mandi_id"].unique())
        mandis_with_price = set(p["mandi_id"].unique())
        missing = mandis_with_arrivals - mandis_with_price
        if not missing or len(mandis_with_arrivals) == 0:
            return None
        share = len(missing) / len(mandis_with_arrivals)
        if share < 0.05:
            return None
        return Finding(
            severity="info",
            title="Some mandis have arrivals but no price records",
            finding=(
                "Market value for these mandis is estimated using the crop's network-wide monthly "
                "average price, so their revenue figures are weaker evidence than the rest."
            ),
            evidence=(
                f"{len(missing)} of {len(mandis_with_arrivals)} mandis with arrivals "
                f"({share*100:.0f}%) have no matching price rows in this window."
            ),
            action="Treat their revenue contribution as indicative; the Data Quality tab lists the fallback rule.",
            rank=25 + share * 20,
        )

    # ---- check 10: peak season -------------------------------------------
    def peak_season():
        a = ctx.arrivals
        if a.empty or "ym" not in a.columns:
            return None
        monthly = _complete_months(a, "arrival_quantity_qtl")
        if len(monthly) < 3:
            return None
        peak = monthly.idxmax()
        ratio = monthly.max() / max(monthly.median(), 1)
        if ratio < 1.3:
            return None
        return Finding(
            severity="info",
            title="Clear arrival peak identified",
            finding=(
                "Arrivals are strongly seasonal in this selection, and peak months are when MSP "
                "breaches typically worsen as supply outruns buying capacity."
            ),
            evidence=(
                f"{peak} is the peak month at {monthly.max():,.0f} Qtl, "
                f"{ratio:.1f}× the median month."
            ),
            action="Staff procurement desks and transport capacity against this peak, not the annual average.",
            rank=30 + ratio * 5,
        )

    for check in (msp_hotspot, crop_exposure, revenue_trend, logistics_bottleneck,
                  inefficient_routes, concentration, volatility, weather_pressure,
                  coverage_gap, peak_season):
        res = _safe(check)
        if res:
            out.append(res)

    out.sort(key=lambda i: (SEVERITY_ORDER.get(i.severity, 9), -i.rank))
    return out


def _mandi_name(ctx: QueryContext, mandi_id) -> str:
    try:
        row = ctx.master.loc[ctx.master["mandi_id"] == mandi_id, "mandi_name"]
        if len(row):
            return str(row.iloc[0])
    except Exception:
        pass
    return str(mandi_id)


# --------------------------------------------------------------------------
# 2. REACTIVE Q&A  (rule-based intent routing over the dataframes)
# --------------------------------------------------------------------------
SUGGESTED_QUESTIONS = [
    "Which mandi is riskiest?",
    "Which crop falls below MSP most?",
    "What is the total revenue?",
    "Which mandi has the highest arrivals?",
    "Which warehouse is slowest?",
    "What is the average price of wheat?",
    "How many farmers were served?",
    "Is revenue trending up or down?",
]


def _find_crop(q: str, ctx: QueryContext):
    """Match a crop name as a whole word.

    Substring matching is unsafe here: 'rice' is inside 'price', so the question
    "what is the average price of wheat" would otherwise match Rice.
    """
    crops = ctx.price["crop_name"].dropna().unique() if not ctx.price.empty else []
    for c in crops:
        if re.search(rf"\b{re.escape(str(c).lower())}\b", q):
            return c
    return None


def answer_question(q: str, ctx: QueryContext) -> dict:
    """Return {'answer': str, 'table': DataFrame|None} for a natural-language question."""
    ql = (q or "").strip().lower()
    if not ql:
        return {"answer": "Ask me something about the current selection.", "table": None}

    crop = _find_crop(ql, ctx)
    m = ctx.metrics

    # --- risk ---------------------------------------------------------------
    if any(k in ql for k in ["riskiest", "highest risk", "most risky", "worst mandi", "risky"]):
        if ctx.risk.empty:
            return {"answer": "No risk scores could be computed for this selection.", "table": None}
        r = ctx.risk.sort_values("score", ascending=False)
        top = r.iloc[0]
        tbl = r.head(5)[["mandi_name", "district", "score"]].rename(
            columns={"mandi_name": "Mandi", "district": "District", "score": "Risk score"})
        tbl["Risk score"] = tbl["Risk score"].round(1)
        return {
            "answer": (
                f"**{top['mandi_name']}** ({top['district']}) carries the highest composite risk score at "
                f"**{top['score']:.0f}/100**, blending below-MSP rate, price volatility and transit delay "
                f"using your current sidebar weights."
            ),
            "table": tbl,
        }

    if any(k in ql for k in ["safest", "lowest risk", "least risky", "best mandi"]):
        if ctx.risk.empty:
            return {"answer": "No risk scores available for this selection.", "table": None}
        r = ctx.risk.sort_values("score")
        top = r.iloc[0]
        tbl = r.head(5)[["mandi_name", "district", "score"]].rename(
            columns={"mandi_name": "Mandi", "district": "District", "score": "Risk score"})
        tbl["Risk score"] = tbl["Risk score"].round(1)
        return {"answer": f"**{top['mandi_name']}** ({top['district']}) is lowest-risk at **{top['score']:.0f}/100**.",
                "table": tbl}

    # --- MSP ---------------------------------------------------------------
    if "msp" in ql or "below" in ql or "shortfall" in ql:
        p = ctx.price.dropna(subset=["msp"]).copy()
        if p.empty:
            return {"answer": "No MSP records in this selection.", "table": None}
        p["below"] = p["modal_price"] < p["msp"]
        if crop:
            sub = p[p["crop_name"] == crop]
            if sub.empty:
                return {"answer": f"No {crop} price records with MSP in this selection.", "table": None}
            rate = sub["below"].mean()
            gap = (sub.loc[sub["below"], "msp"] - sub.loc[sub["below"], "modal_price"]).mean()
            return {
                "answer": (
                    f"**{rate*100:.1f}%** of **{crop}** sales priced below MSP "
                    f"({int(sub['below'].sum())} of {len(sub)} records). "
                    f"When below, the average shortfall is **₹{gap:,.0f}/quintal**."
                ),
                "table": None,
            }
        by_crop = p.groupby("crop_name")["below"].mean().sort_values(ascending=False)
        tbl = (by_crop * 100).round(1).reset_index()
        tbl.columns = ["Crop", "% below MSP"]
        return {
            "answer": (
                f"Overall **{p['below'].mean()*100:.1f}%** of sales cleared below MSP. "
                f"**{by_crop.index[0]}** is worst at **{by_crop.iloc[0]*100:.1f}%**."
            ),
            "table": tbl,
        }

    # --- revenue / value ----------------------------------------------------
    if any(k in ql for k in ["revenue", "market value", "worth", "turnover"]):
        if "trend" in ql or "up or down" in ql or "trending" in ql:
            v = ctx.valued.dropna(subset=["market_value"])
            monthly = _complete_months(v, "market_value")
            if len(monthly) < 3:
                return {"answer": "Not enough complete months in this window to judge a trend.", "table": None}
            slope = np.polyfit(np.arange(len(monthly)), monthly.values, 1)[0]
            pct = 100 * slope / max(monthly.mean(), 1)
            word = "rising" if pct > 0 else "falling"
            return {"answer": f"Market value is **{word}** at roughly **{pct:+.1f}% per month** across {len(monthly)} months.",
                    "table": None}
        rev = m.get("total_revenue", np.nan)
        if crop:
            v = ctx.valued[ctx.valued["crop_name"] == crop]
            cr = v["market_value"].sum()
            share = 100 * cr / rev if rev else np.nan
            return {"answer": f"**{crop}** accounts for **₹{cr/1e7:,.2f} Cr**, about **{share:.1f}%** of the selection's total.",
                    "table": None}
        by_crop = ctx.valued.groupby("crop_name")["market_value"].sum().sort_values(ascending=False)
        tbl = (by_crop / 1e7).round(2).reset_index()
        tbl.columns = ["Crop", "Market value (₹ Cr)"]
        return {"answer": f"Estimated market value for the current selection is **₹{rev/1e7:,.2f} Cr**.", "table": tbl}

    # --- arrivals / volume --------------------------------------------------
    if any(k in ql for k in ["arrival", "volume", "quantity", "busiest", "largest mandi", "highest arrival"]):
        a = ctx.arrivals
        if a.empty:
            return {"answer": "No arrival records in this selection.", "table": None}
        by_mandi = a.groupby("mandi_id")["arrival_quantity_qtl"].sum().sort_values(ascending=False)
        top_id = by_mandi.index[0]
        tbl = by_mandi.head(5).reset_index()
        tbl["mandi_id"] = tbl["mandi_id"].apply(lambda x: _mandi_name(ctx, x))
        tbl.columns = ["Mandi", "Arrivals (Qtl)"]
        tbl["Arrivals (Qtl)"] = tbl["Arrivals (Qtl)"].round(0)
        return {
            "answer": (
                f"**{_mandi_name(ctx, top_id)}** handles the most volume at "
                f"**{by_mandi.iloc[0]:,.0f} Qtl**, out of {a['arrival_quantity_qtl'].sum():,.0f} Qtl total."
            ),
            "table": tbl,
        }

    # --- logistics ----------------------------------------------------------
    if any(k in ql for k in ["warehouse", "transit", "slow", "delay", "logistic", "transport", "distance"]):
        t = ctx.transport.dropna(subset=["clean_transit_hours"])
        if t.empty:
            return {"answer": "No transport records in this selection.", "table": None}
        wh = t.groupby("destination_warehouse")["clean_transit_hours"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        tbl = wh.reset_index()
        tbl.columns = ["Warehouse", "Avg transit (hrs)", "Trips"]
        tbl["Avg transit (hrs)"] = tbl["Avg transit (hrs)"].round(1)
        return {
            "answer": (
                f"**{wh.index[0]}** is slowest at **{wh.iloc[0]['mean']:.1f} hrs** average transit; "
                f"the network average is **{t['clean_transit_hours'].mean():.1f} hrs**."
            ),
            "table": tbl,
        }

    # --- price --------------------------------------------------------------
    if any(k in ql for k in ["price", "rate", "cost", "expensive", "cheap"]):
        p = ctx.price
        if p.empty:
            return {"answer": "No price records in this selection.", "table": None}
        if crop:
            sub = p[p["crop_name"] == crop]
            return {
                "answer": (
                    f"**{crop}** averages **₹{sub['modal_price'].mean():,.0f}/quintal** "
                    f"(range ₹{sub['modal_price'].min():,.0f}–₹{sub['modal_price'].max():,.0f}) "
                    f"across {len(sub):,} records."
                ),
                "table": None,
            }
        by_crop = p.groupby("crop_name")["modal_price"].mean().sort_values(ascending=False)
        tbl = by_crop.round(0).reset_index()
        tbl.columns = ["Crop", "Avg modal price (₹/qtl)"]
        return {"answer": f"Average modal price across all selected crops is **₹{p['modal_price'].mean():,.0f}/quintal**.",
                "table": tbl}

    # --- farmers ------------------------------------------------------------
    if "farmer" in ql:
        a = ctx.arrivals
        if a.empty or "farmer_count" not in a.columns:
            return {"answer": "No farmer-count data in this selection.", "table": None}
        total = a["farmer_count"].sum()
        avg_lot = a["arrival_quantity_qtl"].sum() / max(total, 1)
        return {"answer": f"**{total:,.0f}** farmer visits recorded, averaging **{avg_lot:.2f} Qtl** per farmer.",
                "table": None}

    # --- weather ------------------------------------------------------------
    if any(k in ql for k in ["weather", "rain", "temperature", "heat", "humid"]):
        w = ctx.weather
        if w.empty:
            return {"answer": "No weather readings in this date window.", "table": None}
        return {
            "answer": (
                f"Average temperature **{w['temperature_celsius'].mean():.1f}°C** "
                f"(max {w['temperature_celsius'].max():.1f}°C), average rainfall "
                f"**{w['rainfall_mm'].mean():.1f} mm**, average humidity "
                f"**{w['humidity_percent'].mean():.0f}%** across {len(w):,} readings."
            ),
            "table": None,
        }

    # --- count / how many mandis -------------------------------------------
    if "how many" in ql and "mandi" in ql:
        return {"answer": f"**{ctx.arrivals['mandi_id'].nunique()}** mandis appear in the current selection.", "table": None}

    # --- fallback -----------------------------------------------------------
    return {
        "answer": (
            "I couldn't map that to a metric I compute. Try asking about **risk**, **MSP**, "
            "**revenue**, **arrivals**, **prices**, **farmers**, **logistics** or **weather**. "
            "Or use one of the common questions."
        ),
        "table": None,
    }