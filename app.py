"""
MandiPulse
----------
Streamlit dashboard over five agri-market CSVs (price and MSP, arrivals,
transport, weather sensors, mandi master). The question it tries to answer:
where are farmers getting less than MSP, and what is driving it?

Run with:  streamlit run app.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from insights import QueryContext, SUGGESTED_QUESTIONS, answer_question, find_issues

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MandiPulse",
    page_icon=str(Path(__file__).parent / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# THEME
# Page colours (background, text, sidebar) come from .streamlit/config.toml,
# which defines a light and a dark palette. The CSS below only styles our own
# HTML, and it borrows the current text colour (currentColor) for rules and
# washes, so it follows whichever theme is active without any detection code.
# --------------------------------------------------------------------------
PRIMARY = "#C08A2A"      # ochre: main accent
DANGER = "#C4553A"       # brick: below MSP / high risk
GOOD = "#4C8C6B"         # sage green: healthy / low risk
WARN = PRIMARY           # medium severity shares the ochre
SECONDARY = "#8E8A7E"    # warm grey for neutral series
GRID = "rgba(128,128,128,0.18)"
LINE = "rgba(128,128,128,0.40)"

# Same colour for a crop on every chart in every tab.
CROP_COLORS = {
    "Wheat": "#C9A13B",
    "Rice": "#5E9C84",
    "Maize": "#D4803F",
    "Cotton": "#8D93A6",
    "Mustard": "#B5533C",
    "Sugarcane": "#5B7FA6",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
    --accent: @PRIMARY@;
    --good: @GOOD@;
    --bad: @DANGER@;
    --rule: color-mix(in srgb, currentColor 16%, transparent);
    --rule-strong: color-mix(in srgb, currentColor 65%, transparent);
    --wash: color-mix(in srgb, currentColor 5%, transparent);
    --muted: color-mix(in srgb, currentColor 58%, transparent);
    --serif: 'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif;
}

.block-container { padding-top: 4.5rem; max-width: 1400px; }
h1, h2, h3 { font-family: var(--serif); font-weight: 600; letter-spacing: -0.01em; }

/* ---------- masthead ---------- */
.masthead {
    display: flex; justify-content: space-between; align-items: flex-end;
    flex-wrap: wrap; gap: 14px;
    padding-bottom: 12px; margin-bottom: 26px;
    border-bottom: 4px double var(--rule-strong);
}
.masthead h1 { margin: 0; padding: 0.1em 0 0; font-size: 2.15rem; line-height: 1.2; overflow: visible; }
.mh-sub {
    margin-top: 6px; font-size: 0.78rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
}
.mh-meta { text-align: right; font-family: var(--serif); font-size: 1.0rem; }
.mh-meta span {
    display: block; font-family: inherit; font-size: 0.66rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 2px;
}

/* ---------- key figures ---------- */
.kpi {
    border-top: 2px solid var(--rule-strong);
    padding: 10px 6px 4px 0; min-height: 124px;
}
.kpi.risk { border-top-color: var(--bad); }
.kpi.gold { border-top-color: var(--accent); }
.kpi-label {
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); font-weight: 600;
}
.kpi-value {
    font-family: var(--serif); font-size: 2.1rem; font-weight: 600; line-height: 1.15;
    margin-top: 6px; font-variant-numeric: lining-nums tabular-nums;
}
.kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
.kpi-delta { font-size: 0.78rem; font-weight: 600; margin-top: 8px; font-variant-numeric: tabular-nums; }
.kpi-delta.up-good, .kpi-delta.down-good { color: color-mix(in srgb, var(--good) 70%, currentColor); }
.kpi-delta.up-bad, .kpi-delta.down-bad { color: color-mix(in srgb, var(--bad) 70%, currentColor); }
.kpi-delta.flat { color: var(--muted); }

/* ---------- section headings ---------- */
.section-head {
    display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
    margin: 28px 0 10px 0; padding-bottom: 6px; border-bottom: 1px solid var(--rule);
}
.section-head .title { font-family: var(--serif); font-size: 1.2rem; font-weight: 600; }
.section-head .caption { font-size: 0.8rem; color: var(--muted); font-style: italic; }

.story-box {
    font-family: var(--serif); font-size: 1.15rem; line-height: 1.65;
    padding: 14px 0; margin: 0 0 6px 0;
    border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
}
.story-box b { color: color-mix(in srgb, var(--accent) 80%, currentColor); }

/* ---------- badges ---------- */
.badge {
    display: inline-block; padding: 1px 8px; border-radius: 2px; border: 1px solid;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
}
.badge-high {
    color: color-mix(in srgb, var(--bad) 70%, currentColor);
    background: color-mix(in srgb, var(--bad) 13%, transparent);
    border-color: color-mix(in srgb, var(--bad) 45%, transparent);
}
.badge-med {
    color: color-mix(in srgb, var(--accent) 70%, currentColor);
    background: color-mix(in srgb, var(--accent) 14%, transparent);
    border-color: color-mix(in srgb, var(--accent) 45%, transparent);
}
.badge-low {
    color: color-mix(in srgb, var(--good) 70%, currentColor);
    background: color-mix(in srgb, var(--good) 13%, transparent);
    border-color: color-mix(in srgb, var(--good) 45%, transparent);
}

/* ---------- sidebar ---------- */
.sidebar-brand { padding: 2px 0 12px 0; }
.sidebar-brand .name { font-family: var(--serif); font-size: 1.55rem; font-weight: 600; line-height: 1.1; }
.side-divider {
    border-top: 1px solid var(--rule); margin: 18px 0 8px 0; padding-top: 10px;
    font-size: 0.66rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted); font-weight: 700;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab"] {
    font-family: var(--serif); font-size: 0.98rem; font-weight: 600; padding: 8px 14px;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); height: 3px; }
.stTabs [data-baseweb="tab-border"] { background-color: transparent; }

/* ---------- tables (HTML tables from pandas) ---------- */
.stMarkdown table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.stMarkdown table th {
    text-align: left; font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; padding: 6px 8px; border-bottom: 1px solid var(--rule-strong);
}
.stMarkdown table td { padding: 8px; border-bottom: 1px solid var(--rule); }
.stMarkdown table tr:hover td { background: var(--wash); }

/* ---------- checks list ---------- */
.insight { padding: 14px 0; border-bottom: 1px solid var(--rule); }
.insight .ins-head { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.insight .ins-head::before {
    content: ""; width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: var(--muted);
}
.insight.high .ins-head::before { background: var(--bad); }
.insight.medium .ins-head::before { background: var(--accent); }
.insight.low .ins-head::before { background: var(--good); }
.insight .ins-title { font-family: var(--serif); font-weight: 600; font-size: 1.04rem; }
.insight .ins-body { font-size: 0.9rem; line-height: 1.6; }
.insight .ins-meta { font-size: 0.82rem; color: var(--muted); margin-top: 6px; line-height: 1.5; }
.insight .ins-meta b {
    font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; margin-right: 4px;
}

/* ---------- ask the data ---------- */
.st-key-qa_search_wrap {
    border-bottom: 1px solid var(--rule-strong); padding: 2px 0 2px 2px; margin-bottom: 6px;
}
.st-key-qa_search_wrap:focus-within { border-bottom-color: var(--accent); }
.st-key-qa_search_wrap div[data-testid="stForm"] {
    border: none !important; padding: 0 !important; background: transparent !important;
}
.st-key-qa_search_wrap div[data-baseweb="base-input"],
.st-key-qa_search_wrap div[data-baseweb="input"] > div {
    border: none !important; background: transparent !important; box-shadow: none !important;
}
.qa-label {
    font-size: 0.66rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted); font-weight: 700; margin: 14px 0 8px 0;
}
.st-key-qa_chip_row button {
    border-radius: 2px !important; background: transparent !important;
    border: 1px solid var(--rule) !important; font-size: 0.78rem !important; padding: 6px !important;
}
.st-key-qa_chip_row button:hover { border-color: var(--accent) !important; }
.qa-q { font-family: var(--serif); font-weight: 600; font-size: 1.02rem; margin: 20px 0 4px 0; }
.qa-answer {
    font-size: 0.93rem; line-height: 1.65; padding-bottom: 12px; border-bottom: 1px solid var(--rule);
}

hr.rule { border: none; border-top: 1px solid var(--rule); margin: 24px 0 14px 0; }
.app-footer {
    margin-top: 36px; padding-top: 10px; border-top: 1px solid var(--rule);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;
    font-size: 0.74rem; color: var(--muted);
}

/* ---------- subtle pop-in animation ---------- */
@keyframes pop-in {
    from { opacity: 0; transform: translateY(8px) scale(0.985); }
    to   { opacity: 1; transform: none; }
}
@keyframes fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.kpi, .section-head, .story-box, .insight, .qa-q, .qa-answer {
    animation: pop-in 0.35s cubic-bezier(0.2, 0.7, 0.3, 1) backwards;
}
.stPlotlyChart { animation: fade-in 0.45s ease backwards; }

/* stagger the four key figures */
[data-testid="stColumn"]:nth-child(2) .kpi, [data-testid="column"]:nth-child(2) .kpi { animation-delay: 0.05s; }
[data-testid="stColumn"]:nth-child(3) .kpi, [data-testid="column"]:nth-child(3) .kpi { animation-delay: 0.10s; }
[data-testid="stColumn"]:nth-child(4) .kpi, [data-testid="column"]:nth-child(4) .kpi { animation-delay: 0.15s; }

/* small hover lift on key figures */
.kpi { transition: transform 0.2s ease; }
.kpi:hover { transform: translateY(-2px); }

/* pop-ups: popovers and dialogs ease in */
[data-testid="stPopoverBody"] > * { animation: pop-in 0.22s cubic-bezier(0.2, 0.7, 0.3, 1) backwards; }
div[data-testid="stDialog"] div[role="dialog"] > * { animation: pop-in 0.28s cubic-bezier(0.2, 0.7, 0.3, 1) backwards; }

@media (prefers-reduced-motion: reduce) {
    .kpi, .section-head, .story-box, .insight, .qa-q, .qa-answer, .stPlotlyChart,
    [data-testid="stPopoverBody"] > *, div[data-testid="stDialog"] div[role="dialog"] > * { animation: none; }
}
</style>
"""
for _k, _v in {"@PRIMARY@": PRIMARY, "@GOOD@": GOOD, "@DANGER@": DANGER}.items():
    _CSS = _CSS.replace(_k, _v)
st.markdown(_CSS, unsafe_allow_html=True)

def inr(n, decimals=0):
    """Indian digit grouping: 1234567 -> 12,34,567."""
    if n is None or pd.isna(n):
        return "N/A"
    neg = n < 0
    whole, _, frac = f"{abs(n):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail, parts = whole[:-3], whole[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + ("." + frac if frac else "")


def kpi_card(label, value, sub="", kind="", delta=None, delta_good_when="up"):
    """delta: (pct_change, direction 'up'/'down') or None. delta_good_when: 'up' means
    an increase is favourable (green), 'down' means a decrease is favourable (e.g. risk)."""
    cls = f"kpi {kind}".strip()
    delta_html = ""
    if delta is not None:
        pct, direction = delta
        favourable = (direction == delta_good_when)
        arrow = "▲" if direction == "up" else "▼"
        tone = ("up-good" if (direction == "up" and favourable) else
                "down-good" if (direction == "down" and favourable) else
                "up-bad" if direction == "up" else "down-bad")
        delta_html = f'<div class="kpi-delta {tone}">{arrow} {abs(pct):,.1f}% vs prior period</div>'
    else:
        delta_html = '<div class="kpi-delta flat">no earlier period to compare</div>'
    st.markdown(
        f"""<div class="{cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
                {delta_html}
            </div>""",
        unsafe_allow_html=True,
    )


def section_head(title, caption=""):
    st.markdown(
        f"""<div class="section-head">
            <span class="title">{title}</span>
            <span class="caption">{caption}</span></div>""",
        unsafe_allow_html=True,
    )


SEV_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}


def risk_badge(score):
    if score >= 66:
        return '<span class="badge badge-high">High risk</span>'
    if score >= 33:
        return '<span class="badge badge-med">Medium risk</span>'
    return '<span class="badge badge-low">Low risk</span>'


CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Sans Pro, -apple-system, Segoe UI, sans-serif", size=12),
    margin=dict(t=30, l=8, r=8, b=8),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor=GRID, linecolor=LINE, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=LINE, zerolinecolor=GRID),
)

# Sand to brick, used for every risk / intensity scale
RISK_RAMP = ["#D9CDB0", "#D49A5A", DANGER]


def show_chart(fig, **kwargs):
    """Draw a plotly figure. Figures without a title get an empty one, because
    newer plotly.js prints the word "undefined" when a title object has no text."""
    if fig.layout.title.text is None:
        fig.update_layout(title_text="")
    st.plotly_chart(fig, **kwargs)


# --------------------------------------------------------------------------
# DATA LOADING + CLEANING  (cached; raw CSVs in, tidy frames out)
# --------------------------------------------------------------------------
def normalize_mandi_id(x):
    """MANDI-042 / mandi_019 / '038' / MANDI050  ->  MANDI_038 (one format)."""
    if pd.isna(x):
        return np.nan
    digits = re.findall(r"\d+", str(x))
    if not digits:
        return np.nan
    return f"MANDI_{digits[0].zfill(3)}"


@st.cache_data(show_spinner=True)
def load_data():
    quality_log = {}

    # ---- mandi master -----------------------------------------------
    master = pd.read_csv(DATA_DIR / "clean_mandi_master.csv")
    master["mandi_id"] = master["mandi_id"].apply(normalize_mandi_id)
    master["mandi_type"] = (
        master["mandi_type"].astype(str).str.strip().str.upper()
        .replace({"NAN": np.nan})
        .map({"APMC": "APMC", "PRIVATE": "Private", "DIRECT": "Direct"})
    )
    master["district"] = master["district"].astype(str).str.strip().str.title().replace("Nan", np.nan)
    master["state"] = master["state"].astype(str).str.strip().replace("nan", np.nan)
    master = master.drop_duplicates(subset="mandi_id")

    # ---- price & MSP ---------------------------------------------------
    price_raw = pd.read_csv(DATA_DIR / "clean_price_and_msp.csv")
    price = price_raw.copy()
    price["mandi_id"] = price["mandi_id"].apply(normalize_mandi_id)
    price["clean_date"] = pd.to_datetime(price["clean_date"], errors="coerce")
    n_before = len(price)
    # Values under 50 are unit/decimal entry errors for these crops (real
    # modal prices run into the thousands per quintal) -> treat as missing.
    for c in ["min_price", "max_price", "modal_price", "msp"]:
        bad = price[c] < 50
        price.loc[bad, c] = np.nan
    price = price.dropna(subset=["clean_date", "modal_price", "mandi_id"])
    quality_log["price"] = dict(
        raw_rows=n_before,
        clean_rows=len(price),
        dropped_pct=round(100 * (1 - len(price) / n_before), 1),
    )

    # ---- arrivals --------------------------------------------------------
    arr_raw = pd.read_csv(DATA_DIR / "clean_mandi_arrivals.csv")
    arr = arr_raw.copy()
    arr["mandi_id"] = arr["mandi_id"].apply(normalize_mandi_id)
    arr["clean_date"] = pd.to_datetime(arr["clean_date"], errors="coerce")
    n_before = len(arr)
    arr["arrival_quantity_qtl"] = arr["arrival_quantity_qtl"].abs()  # fix sign errors
    arr = arr.dropna(subset=["clean_date", "mandi_id", "arrival_quantity_qtl"])
    quality_log["arrivals"] = dict(
        raw_rows=n_before,
        clean_rows=len(arr),
        dropped_pct=round(100 * (1 - len(arr) / n_before), 1),
    )

    # ---- transport logistics ---------------------------------------------
    trans_raw = pd.read_csv(DATA_DIR / "clean_transport_logistics.csv")
    trans = trans_raw.copy()
    trans["mandi_id"] = trans["mandi_id"].apply(normalize_mandi_id)
    trans["clean_date"] = pd.to_datetime(trans["clean_date"], errors="coerce")
    trans["distance_km"] = pd.to_numeric(trans["distance_km"], errors="coerce")
    n_before = len(trans)
    # transit-hour parsing occasionally leaked a 4-digit year into the field
    trans.loc[trans["clean_transit_hours"] > 200, "clean_transit_hours"] = np.nan
    trans = trans.dropna(subset=["mandi_id", "clean_transit_hours"])
    quality_log["transport"] = dict(
        raw_rows=n_before,
        clean_rows=len(trans),
        dropped_pct=round(100 * (1 - len(trans) / n_before), 1),
    )

    # ---- weather -----------------------------------------------------------
    weather_raw = pd.read_csv(DATA_DIR / "clean_weather_sensors.csv")
    weather = weather_raw.copy()
    weather["clean_date"] = pd.to_datetime(weather["clean_date"], errors="coerce")
    n_before = len(weather)
    weather = weather.dropna(subset=["clean_date"])
    quality_log["weather"] = dict(
        raw_rows=n_before,
        clean_rows=len(weather),
        dropped_pct=round(100 * (1 - len(weather) / n_before), 1),
    )

    # ---- enrich price / arrivals / transport with master geography -------
    # (master is the canonical mandi -> geography map; drop each file's own
    # loosely-formatted district field in favour of it for consistency)
    price = price.drop(columns=["district"], errors="ignore").merge(
        master[["mandi_id", "state", "district", "mandi_type"]], on="mandi_id", how="left"
    )
    arr = arr.merge(master[["mandi_id", "state", "district", "mandi_type"]], on="mandi_id", how="left")
    trans = trans.merge(master[["mandi_id", "state", "district", "mandi_type"]], on="mandi_id", how="left")

    return master, price, arr, trans, weather, quality_log


master, price, arr, trans, weather, quality_log = load_data()

# --------------------------------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------------------------------
st.sidebar.markdown(
    """<div class="sidebar-brand">
        <div><div class="name">MandiPulse</div></div>
       </div>""",
    unsafe_allow_html=True,
)

FILTER_KEYS = ["f_date", "f_state", "f_district", "f_type", "f_crop", "f_mandi", "f_wp", "f_wv", "f_wl"]
if st.sidebar.button("Reset filters", width="stretch"):
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)
    st.rerun()

ALL = " ALL"


def multiselect_all(label, options, key):
    """Multiselect with a single-click 'ALL' entry at the top.

    Picking ALL selects every option at once. The widget defaults to ALL, so the
    dashboard opens on the full dataset. Stale selections (e.g. districts that
    disappear after changing the State filter) are dropped automatically so the
    widget never errors on a value that is no longer a valid option.
    """
    options = list(options)
    choices = [ALL] + options

    # Drop any previously-selected values that are no longer valid options.
    if key in st.session_state:
        kept = [v for v in st.session_state[key] if v in choices]
        st.session_state[key] = kept if kept else [ALL]
    else:
        st.session_state[key] = [ALL]

    picked = st.sidebar.multiselect(label, choices, key=key)

    if ALL in picked or not picked:
        return options
    return picked


st.sidebar.markdown('<div class="side-divider">Scope</div>', unsafe_allow_html=True)

min_date = min(price["clean_date"].min(), arr["clean_date"].min()).date()
max_date = max(price["clean_date"].max(), arr["clean_date"].max()).date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="f_date")
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

states = sorted([s for s in master["state"].dropna().unique()])
sel_states = multiselect_all("State", states, "f_state")

districts_pool = sorted(master.loc[master["state"].isin(sel_states), "district"].dropna().unique())
sel_districts = multiselect_all("District", districts_pool, "f_district")

mandi_types = sorted(master["mandi_type"].dropna().unique())
sel_types = multiselect_all("Mandi type", mandi_types, "f_type")

crops = sorted(price["crop_name"].dropna().unique())
sel_crops = multiselect_all("Crop", crops, "f_crop")

eligible_mandis = sorted(
    master.loc[
        master["state"].isin(sel_states)
        & master["district"].isin(sel_districts)
        & master["mandi_type"].isin(sel_types),
        "mandi_id",
    ].unique()
)
sel_mandis = multiselect_all("Mandi", eligible_mandis, "f_mandi")

st.sidebar.markdown('<div class="side-divider">Risk score</div>', unsafe_allow_html=True)
with st.sidebar.expander("Adjust weights"):
    st.caption("Weights used for the Mandi Risk Score on the Overview tab.")
    w_price = st.slider("Price / MSP weight", 0, 100, 45, key="f_wp")
    w_vol = st.slider("Price volatility weight", 0, 100, 25, key="f_wv")
    w_log = st.slider("Logistics delay weight", 0, 100, 30, key="f_wl")
w_total = max(w_price + w_vol + w_log, 1)



def in_filters(df):
    m = (
        df["clean_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        & df["mandi_id"].isin(sel_mandis)
    )
    if "crop_name" in df.columns:
        m &= df["crop_name"].isin(sel_crops)
    return df[m]


f_price = in_filters(price)
f_arr = in_filters(arr)
f_trans = in_filters(trans)

if f_price.empty or f_arr.empty:
    st.warning("No data matches the current filters. Widen the date range or the selection.")
    st.stop()

# --------------------------------------------------------------------------
# CORE METRIC CALCULATIONS
# --------------------------------------------------------------------------
f_price = f_price.copy()
f_arr = f_arr.copy()
f_price["ym"] = f_price["clean_date"].dt.to_period("M")
f_arr["ym"] = f_arr["clean_date"].dt.to_period("M")

mandi_month_price = f_price.groupby(["mandi_id", "crop_name", "ym"])["modal_price"].mean().reset_index()
crop_month_price = f_price.groupby(["crop_name", "ym"])["modal_price"].mean().reset_index().rename(
    columns={"modal_price": "crop_modal_price"}
)

valued = f_arr.merge(mandi_month_price, on=["mandi_id", "crop_name", "ym"], how="left")
valued = valued.merge(crop_month_price, on=["crop_name", "ym"], how="left")
valued["modal_price"] = valued["modal_price"].fillna(valued["crop_modal_price"])
valued["market_value"] = valued["arrival_quantity_qtl"] * valued["modal_price"]

total_revenue = valued["market_value"].sum()
total_arrivals = f_arr["arrival_quantity_qtl"].sum()

msp_scope = f_price.dropna(subset=["msp"]).copy()
msp_scope["below_msp"] = msp_scope["modal_price"] < msp_scope["msp"]
below_msp_rate = msp_scope["below_msp"].mean() if len(msp_scope) else np.nan
msp_compliance_rate = 1 - below_msp_rate if pd.notna(below_msp_rate) else np.nan

avg_transit = f_trans["clean_transit_hours"].mean() if len(f_trans) else np.nan
avg_distance = f_trans["distance_km"].mean() if len(f_trans) else np.nan

w_in_range = weather[weather["clean_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
if len(w_in_range):
    heat_days = (w_in_range["temperature_celsius"] > 35).mean()
    heavy_rain_days = (w_in_range["rainfall_mm"] > w_in_range["rainfall_mm"].quantile(0.85)).mean()
    weather_risk_index = 100 * np.nanmean([heat_days, heavy_rain_days])
else:
    weather_risk_index = np.nan

# ---- period-over-period comparison (same-length window immediately before) ----
period_days = (end_date - start_date).days + 1
prev_end = pd.Timestamp(start_date) - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days - 1)

def prev_period(df):
    m = df["clean_date"].between(prev_start, prev_end) & df["mandi_id"].isin(sel_mandis)
    if "crop_name" in df.columns:
        m &= df["crop_name"].isin(sel_crops)
    return df[m]

p_price = prev_period(price)
p_arr = prev_period(arr)
p_trans = prev_period(trans)

def pct_delta(curr, prev):
    if prev is None or pd.isna(prev) or prev == 0 or pd.isna(curr):
        return None
    change = 100 * (curr - prev) / prev
    return (change, "up" if change >= 0 else "down")

if len(p_arr) and len(p_price):
    p_price_ = p_price.copy()
    p_arr_ = p_arr.copy()
    p_price_["ym"] = p_price_["clean_date"].dt.to_period("M")
    p_arr_["ym"] = p_arr_["clean_date"].dt.to_period("M")
    p_mmp = p_price_.groupby(["mandi_id", "crop_name", "ym"])["modal_price"].mean().reset_index()
    p_cmp = p_price_.groupby(["crop_name", "ym"])["modal_price"].mean().reset_index().rename(columns={"modal_price": "crop_modal_price"})
    p_valued = p_arr_.merge(p_mmp, on=["mandi_id", "crop_name", "ym"], how="left").merge(p_cmp, on=["crop_name", "ym"], how="left")
    p_valued["modal_price"] = p_valued["modal_price"].fillna(p_valued["crop_modal_price"])
    prev_revenue = (p_valued["arrival_quantity_qtl"] * p_valued["modal_price"]).sum()
    prev_arrivals = p_arr_["arrival_quantity_qtl"].sum()
    p_msp = p_price_.dropna(subset=["msp"]).copy()
    p_msp["below_msp"] = p_msp["modal_price"] < p_msp["msp"]
    prev_below_msp = p_msp["below_msp"].mean() if len(p_msp) else np.nan
else:
    prev_revenue = prev_arrivals = prev_below_msp = np.nan

prev_transit = p_trans["clean_transit_hours"].mean() if len(p_trans) else np.nan

delta_revenue = pct_delta(total_revenue, prev_revenue)
delta_arrivals = pct_delta(total_arrivals, prev_arrivals)
delta_below_msp = pct_delta(below_msp_rate * 100 if pd.notna(below_msp_rate) else np.nan, prev_below_msp * 100 if pd.notna(prev_below_msp) else np.nan)
prev_compliance = (1 - prev_below_msp) * 100 if pd.notna(prev_below_msp) else np.nan
delta_compliance = pct_delta(msp_compliance_rate * 100 if pd.notna(msp_compliance_rate) else np.nan, prev_compliance)
delta_transit = pct_delta(avg_transit, prev_transit)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown(
    f"""<div class="masthead">
        <div>
            <h1>MandiPulse</h1>
            <div class="mh-sub">Where are farmers selling below MSP, and what is behind it?</div>
        </div>
        <div class="mh-meta"><span>Period</span>{start_date:%d %b %Y} &ndash; {end_date:%d %b %Y}</div>
       </div>""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Est. market value", f"₹{total_revenue/1e7:,.1f} Cr", f"across {len(sel_mandis)} mandis",
             delta=delta_revenue, delta_good_when="up")
    with st.popover("Breakdown"):
        _bd = (valued.groupby("crop_name")["market_value"].sum().sort_values(ascending=False) / 1e7).round(2)
        st.caption("Market value by crop (₹ Cr)")
        st.dataframe(_bd.rename("₹ Cr").reset_index().rename(columns={"crop_name": "Crop"}), hide_index=True)
with k2:
    kpi_card("Arrivals", f"{inr(total_arrivals)} qtl", f"{f_arr['crop_name'].nunique()} crops in scope",
             delta=delta_arrivals, delta_good_when="up")
    with st.popover("Breakdown"):
        _bd = f_arr.groupby("crop_name")["arrival_quantity_qtl"].sum().sort_values(ascending=False).round(0)
        st.caption("Arrivals by crop (qtl)")
        st.dataframe(_bd.rename("Qtl").reset_index().rename(columns={"crop_name": "Crop"}), hide_index=True)
with k3:
    kpi_card("Sales below MSP", f"{below_msp_rate*100:,.1f}%" if pd.notna(below_msp_rate) else "N/A",
             "share of recorded sales", kind="risk", delta=delta_below_msp, delta_good_when="down")
    with st.popover("Breakdown"):
        _bd = (msp_scope.groupby("crop_name")["below_msp"].mean().sort_values(ascending=False) * 100).round(1)
        st.caption("Share of sales below MSP, by crop (%)")
        st.dataframe(_bd.rename("% below MSP").reset_index().rename(columns={"crop_name": "Crop"}), hide_index=True)
with k4:
    kpi_card("Avg transit time", f"{avg_transit:,.1f} hrs" if pd.notna(avg_transit) else "N/A",
             f"avg trip about {avg_distance:,.0f} km", kind="gold", delta=delta_transit, delta_good_when="down")
    with st.popover("Breakdown"):
        _bd = f_trans.groupby("destination_warehouse")["clean_transit_hours"].mean().sort_values(ascending=False).round(1)
        st.caption("Average transit hours, by warehouse")
        st.dataframe(_bd.rename("Avg hrs").reset_index().rename(columns={"destination_warehouse": "Warehouse"}), hide_index=True)

st.write("")

# --------------------------------------------------------------------------
# COMPOSITE MANDI RISK SCORE  (used by the Overview tab and the Ask tab)
# --------------------------------------------------------------------------
price_risk = msp_scope[msp_scope["mandi_id"].isin(sel_mandis)].groupby("mandi_id")["below_msp"].mean()
vol = f_price.groupby("mandi_id")["modal_price"].agg(["mean", "std"])
vol["cv"] = (vol["std"] / vol["mean"]).fillna(0)
log_risk = f_trans.groupby("mandi_id")["clean_transit_hours"].mean()

risk_df = pd.DataFrame(index=sel_mandis)
risk_df["price_risk"] = price_risk
risk_df["volatility"] = vol["cv"]
risk_df["transit_hrs"] = log_risk
risk_df = risk_df.dropna(how="all")


def _norm(s):
    s = s.astype(float)
    if s.max() == s.min():
        return s * 0
    return (s - s.min()) / (s.max() - s.min())


risk_df["score"] = (
    _norm(risk_df["price_risk"].fillna(risk_df["price_risk"].median())) * (w_price / w_total)
    + _norm(risk_df["volatility"].fillna(risk_df["volatility"].median())) * (w_vol / w_total)
    + _norm(risk_df["transit_hrs"].fillna(risk_df["transit_hrs"].median())) * (w_log / w_total)
) * 100
risk_df = risk_df.merge(master[["mandi_id", "mandi_name", "district", "state"]], left_index=True, right_on="mandi_id")
risk_df = risk_df.sort_values("score", ascending=False)

# --------------------------------------------------------------------------
# CONTEXT for the Ask tab (filtered view only)
# --------------------------------------------------------------------------
ctx = QueryContext(
    price=f_price, arrivals=f_arr, transport=f_trans, weather=w_in_range,
    valued=valued, risk=risk_df, master=master,
    metrics=dict(
        total_revenue=total_revenue, total_arrivals=total_arrivals,
        below_msp_rate=below_msp_rate, avg_transit=avg_transit,
        start=f"{start_date:%d %b %Y}", end=f"{end_date:%d %b %Y}",
    ),
)

# --------------------------------------------------------------------------
# POP-UP: full mandi profile (opened from the Overview tab)
# --------------------------------------------------------------------------
@st.dialog("Mandi profile", width="large")
def mandi_profile(mandi_name):
    r = risk_df[risk_df["mandi_name"] == mandi_name].iloc[0]
    mid = r["mandi_id"]
    st.markdown(
        f"**{mandi_name}** · {r['district']}, {r['state']} &nbsp; {risk_badge(r['score'])}",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", f"{r['score']:,.0f} / 100")
    m2.metric("Below-MSP share", f"{r['price_risk']*100:,.1f}%" if pd.notna(r["price_risk"]) else "N/A")
    m3.metric("Avg transit", f"{r['transit_hrs']:,.1f} hrs" if pd.notna(r["transit_hrs"]) else "N/A")

    view = st.radio("Show", ["Price trend", "Arrivals"], horizontal=True, key="mp_view")
    if view == "Price trend":
        d = f_price[f_price["mandi_id"] == mid].groupby(["ym", "crop_name"])["modal_price"].mean().reset_index()
        if d.empty:
            st.info("No price records for this mandi in the current selection.")
        else:
            d["ym_ts"] = d["ym"].dt.to_timestamp()
            f = px.line(d, x="ym_ts", y="modal_price", color="crop_name", markers=True,
                        labels={"ym_ts": "", "modal_price": "Modal price (₹/qtl)", "crop_name": "Crop"},
                        color_discrete_map=CROP_COLORS)
            f.update_layout(**CHART_TEMPLATE, height=340, hovermode="x unified")
            show_chart(f, width='stretch')
    else:
        d = f_arr[f_arr["mandi_id"] == mid].groupby(["ym", "crop_name"])["arrival_quantity_qtl"].sum().reset_index()
        if d.empty:
            st.info("No arrival records for this mandi in the current selection.")
        else:
            d["ym_ts"] = d["ym"].dt.to_timestamp()
            f = px.bar(d, x="ym_ts", y="arrival_quantity_qtl", color="crop_name",
                       labels={"ym_ts": "", "arrival_quantity_qtl": "Arrivals (Qtl)", "crop_name": "Crop"},
                       color_discrete_map=CROP_COLORS)
            f.update_layout(**CHART_TEMPLATE, height=340)
            show_chart(f, width='stretch')


# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab_story, tab_ask, tab_price, tab_supply, tab_logistics, tab_weather, tab_quality = st.tabs(
    ["Overview", "Questions", "Price & MSP", "Arrivals & Supply", "Logistics", "Weather", "Data Quality"]
)

# ==========================================================================
# TAB 1: OVERVIEW
# ==========================================================================
with tab_story:
    _crop_rates = msp_scope.groupby("crop_name")["below_msp"].mean().sort_values(ascending=False)
    _story = (f"<b>{below_msp_rate*100:,.1f}%</b> of recorded sales in this selection were priced below MSP."
              if pd.notna(below_msp_rate) else "No MSP-linked sales in this selection.")
    if len(_crop_rates) and _crop_rates.iloc[0] > 0:
        _story += f" {_crop_rates.index[0]} is hit hardest ({_crop_rates.iloc[0]*100:,.0f}% of its sales)."
    if len(risk_df):
        _top = risk_df.iloc[0]
        _story += f" {_top['mandi_name']} in {_top['district']} has the highest risk score ({_top['score']:,.0f}/100)."
    st.markdown(f'<div class="story-box">{_story}</div>', unsafe_allow_html=True)

    with st.popover("How is the risk score built?"):
        st.caption("Current weights (change them in the sidebar under Risk score)")
        st.progress(w_price / w_total, text=f"Price / MSP: {w_price / w_total * 100:.0f}%")
        st.progress(w_vol / w_total, text=f"Price volatility: {w_vol / w_total * 100:.0f}%")
        st.progress(w_log / w_total, text=f"Logistics delay: {w_log / w_total * 100:.0f}%")
        st.caption("Each part is scaled 0 to 1 across the mandis in scope, then combined using these weights.")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        section_head("Riskiest mandis", "weighted composite score")
        top_risk = risk_df.head(12).sort_values("score")
        fig = px.bar(
            top_risk, x="score", y="mandi_name", orientation="h",
            color="score", color_continuous_scale=RISK_RAMP,
            labels={"score": "Risk score (0-100)", "mandi_name": ""},
        )
        fig.update_layout(coloraxis_showscale=False, **CHART_TEMPLATE, height=420)
        show_chart(fig, width='stretch')
    with c2:
        section_head("Mandi comparison", "against the network average")
        pick = st.selectbox("Inspect a mandi", risk_df["mandi_name"].tolist(), label_visibility="collapsed")
        row = risk_df[risk_df["mandi_name"] == pick].iloc[0]
        avg_row = risk_df[["price_risk", "volatility", "transit_hrs"]].mean()
        max_hrs = max(risk_df["transit_hrs"].max(), 1)
        cats = ["Below-MSP share", "Price volatility", "Transit (vs slowest)"]
        mine = [row["price_risk"] or 0, row["volatility"] or 0, (row["transit_hrs"] or 0) / max_hrs]
        netw = [avg_row["price_risk"] or 0, avg_row["volatility"] or 0, (avg_row["transit_hrs"] or 0) / max_hrs]
        cmp_fig = go.Figure()
        cmp_fig.add_trace(go.Bar(y=cats, x=mine, name=pick, orientation="h", marker_color=DANGER))
        cmp_fig.add_trace(go.Bar(y=cats, x=netw, name="Network average", orientation="h", marker_color=SECONDARY))
        cmp_fig.update_layout(**{k: v for k, v in CHART_TEMPLATE.items() if k not in ("xaxis", "yaxis")},
                              barmode="group", height=300,
                              xaxis=dict(gridcolor=GRID, range=[0, 1], title="0 = best, 1 = worst in scope"),
                              yaxis=dict(autorange="reversed"))
        show_chart(cmp_fig, width='stretch')
        badge = risk_badge(row["score"])
        st.markdown(f"**{pick}**, {row['district']}, {row['state']}   {badge}   ({row['score']:,.0f}/100)", unsafe_allow_html=True)
        if st.button("Open full profile", key="open_mandi_profile"):
            mandi_profile(pick)

    # ---- risk leaderboard table (riskiest 5 vs safest 5, side by side) ----
    section_head("Risk leaderboard", "riskiest vs. safest mandis in scope")
    lb1, lb2 = st.columns(2)
    with lb1:
        st.caption("Highest risk")
        worst = risk_df.head(5)[["mandi_name", "district", "score"]].copy()
        worst["Risk"] = worst["score"].apply(risk_badge)
        worst = worst.rename(columns={"mandi_name": "Mandi", "district": "District", "score": "Score"})
        worst["Score"] = worst["Score"].round(0).astype(int)
        st.write(worst.to_html(escape=False, index=False), unsafe_allow_html=True)
    with lb2:
        st.caption("Lowest risk")
        best = risk_df.tail(5).sort_values("score")[["mandi_name", "district", "score"]].copy()
        best["Risk"] = best["score"].apply(risk_badge)
        best = best.rename(columns={"mandi_name": "Mandi", "district": "District", "score": "Score"})
        best["Score"] = best["Score"].round(0).astype(int)
        st.write(best.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ---- monthly market value with a straight-line fit over the actual months ----
    monthly_rev = valued.groupby("ym")["market_value"].sum().reset_index()
    monthly_rev["ym_ts"] = monthly_rev["ym"].dt.to_timestamp()
    if len(monthly_rev) >= 3:
        _peak = monthly_rev.loc[monthly_rev["market_value"].idxmax()]
        section_head(f"Market value peaked in {_peak['ym_ts']:%b %Y}", "estimated, by month")
        x = np.arange(len(monthly_rev))
        coeffs = np.polyfit(x, monthly_rev["market_value"], 1)
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Bar(x=monthly_rev["ym_ts"], y=monthly_rev["market_value"], name="Market value", marker_color=PRIMARY))
        trend_fig.add_trace(go.Scatter(x=monthly_rev["ym_ts"], y=np.polyval(coeffs, x), name="Straight-line fit",
                                       mode="lines", line=dict(color=SECONDARY, dash="dash")))
        trend_fig.add_annotation(x=_peak["ym_ts"], y=_peak["market_value"], text="peak", showarrow=True,
                                 arrowhead=0, ax=0, ay=-28)
        trend_fig.update_layout(**CHART_TEMPLATE, height=340, yaxis_title="₹ market value")
        show_chart(trend_fig, width='stretch')
        direction = "up" if coeffs[0] > 0 else "down"
        note = f"The fitted line slopes {direction} across the selected months."
        if valued["clean_date"].max().day < 25:
            note += " The last month is only partly covered by the data."
        st.caption(note)
    else:
        section_head("Market value by month", "")
        st.info("Select a wider date range to see the monthly trend.")

# ==========================================================================
# TAB: ASK THE DATA
# ==========================================================================
with tab_ask:
    if "qa_chat" not in st.session_state:
        st.session_state.qa_chat = []
    if "pending_q" not in st.session_state:
        st.session_state.pending_q = None

    st.subheader("Questions")
    st.caption("Answers are calculated from the filtered data. Try risk, MSP, market value, "
               "arrivals, prices, logistics or weather.")

    with st.container(key="qa_search_wrap"):
        with st.form(key="qa_form", clear_on_submit=True):
            sc1, sc2 = st.columns([9, 1.3])
            with sc1:
                typed = st.text_input(
                    "Question",
                    placeholder="Which crop has the most sales below MSP?",
                    label_visibility="collapsed",
                    key="qa_text",
                )
            with sc2:
                submitted = st.form_submit_button("Ask", width="stretch")

    st.markdown('<div class="qa-label">Common questions</div>', unsafe_allow_html=True)
    with st.container(key="qa_chip_row"):
        qcols = st.columns(4)
        for idx, sq in enumerate(SUGGESTED_QUESTIONS):
            if qcols[idx % 4].button(sq, key=f"sq_{idx}", width="stretch"):
                st.session_state.pending_q = sq

    question = (typed.strip() if submitted and typed and typed.strip() else None) or st.session_state.pending_q
    st.session_state.pending_q = None

    if question:
        result = answer_question(question, ctx)
        st.session_state.qa_chat.insert(0, {"q": question, "a": result["answer"], "table": result["table"]})

    if st.session_state.qa_chat:
        hcol1, hcol2 = st.columns([6, 1])
        with hcol1:
            st.markdown('<div class="qa-label">Previous questions</div>', unsafe_allow_html=True)
        with hcol2:
            if st.button("Clear", key="clear_chat", width="stretch"):
                st.session_state.qa_chat = []
                st.rerun()
        for entry in st.session_state.qa_chat[:12]:
            st.markdown(f'<div class="qa-q">{entry["q"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="qa-answer">{entry["a"]}</div>', unsafe_allow_html=True)
            if entry["table"] is not None and not entry["table"].empty:
                st.dataframe(entry["table"], width='stretch', hide_index=True)
    else:
        st.caption("Type a question above, or pick one from the list.")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ---------------- rule-based checks ----------------
    f1, f2 = st.columns([3, 1])
    with f1:
        section_head("Checks on this selection", "recalculated when the filters change")
    with f2:
        st.write("")
        st.button("Refresh", width="stretch")

    with st.spinner("Running checks..."):
        findings = find_issues(ctx)

    if not findings:
        st.info("No issues found for this selection. Try widening the filters.")
    else:
        sev_counts = {}
        for i in findings:
            sev_counts[i.severity] = sev_counts.get(i.severity, 0) + 1
        st.markdown(
            " ".join(
                f'<span class="badge badge-{"high" if s=="high" else "med" if s=="medium" else "low"}">'
                f'{n} {s}</span>'
                for s, n in sorted(sev_counts.items(), key=lambda kv: SEV_RANK.get(kv[0], 9))
            ),
            unsafe_allow_html=True,
        )
        st.write("")
        for ins in findings:
            st.markdown(
                f"""<div class="insight {ins.severity}">
                    <div class="ins-head">
                        <span class="ins-title">{ins.title}</span>
                        <span class="badge badge-{"high" if ins.severity=="high" else "med" if ins.severity=="medium" else "low"}">{ins.severity}</span>
                    </div>
                    <div class="ins-body">{ins.finding}</div>
                    <div class="ins-meta"><b>Basis:</b> {ins.evidence}</div>
                    <div class="ins-meta"><b>Next step:</b> {ins.action}</div>
                   </div>""",
                unsafe_allow_html=True,
            )


# ==========================================================================
# TAB 2: PRICE & MSP
# ==========================================================================
with tab_price:
    with st.popover("Crop snapshot"):
        snap_crop = st.selectbox("Crop", sorted(f_price["crop_name"].dropna().unique()), key="snap_crop")
        _sp = f_price[f_price["crop_name"] == snap_crop]
        _sm = msp_scope[msp_scope["crop_name"] == snap_crop]
        s1, s2 = st.columns(2)
        s1.metric("Avg modal price", f"₹{inr(_sp['modal_price'].mean())}")
        s2.metric("Avg MSP", f"₹{inr(_sm['msp'].mean())}" if len(_sm) else "N/A")
        s3, s4 = st.columns(2)
        s3.metric("Sales below MSP", f"{_sm['below_msp'].mean()*100:,.1f}%" if len(_sm) else "N/A")
        s4.metric("Price range", f"₹{inr(_sp['modal_price'].min())} to ₹{inr(_sp['modal_price'].max())}")

    left, right = st.columns([1.4, 1])
    with left:
        section_head("Modal price trend", "by crop, monthly average")
        trend = f_price.groupby(["ym", "crop_name"])["modal_price"].mean().reset_index()
        trend["ym_ts"] = trend["ym"].dt.to_timestamp()
        fig = px.line(trend, x="ym_ts", y="modal_price", color="crop_name", markers=True,
                       labels={"ym_ts": "", "modal_price": "Modal price (₹/qtl)", "crop_name": "Crop"},
                       color_discrete_map=CROP_COLORS)
        fig.update_layout(**CHART_TEMPLATE, height=380)
        show_chart(fig, width='stretch')
    with right:
        comp = msp_scope[msp_scope["mandi_id"].isin(sel_mandis)].groupby("crop_name")["below_msp"].mean().sort_values(ascending=False).reset_index()
        comp["below_msp"] *= 100
        section_head(f"{comp.iloc[0]['crop_name']} is hit hardest" if len(comp) else "MSP shortfall rate",
                     "share of sales below MSP, by crop")
        fig2 = px.bar(comp, x="below_msp", y="crop_name", orientation="h",
                       labels={"below_msp": "% sales below MSP", "crop_name": ""},
                       color_discrete_sequence=[SECONDARY])
        fig2.update_traces(marker_color=[DANGER] + [SECONDARY] * (len(comp) - 1))
        fig2.update_layout(**CHART_TEMPLATE, height=380)
        show_chart(fig2, width='stretch')

    section_head("Price spread", "min / modal / max vs MSP, by crop")
    box_df = f_price.melt(id_vars=["crop_name"], value_vars=["min_price", "modal_price", "max_price", "msp"],
                           var_name="metric", value_name="value")
    fig3 = px.box(box_df, x="crop_name", y="value", color="metric", points=False,
                   color_discrete_map={"min_price": "#CDBF9E", "modal_price": PRIMARY, "max_price": "#7F7A6C", "msp": DANGER})
    fig3.update_layout(**CHART_TEMPLATE, height=420, yaxis_title="₹ / quintal")
    show_chart(fig3, width='stretch')

# ==========================================================================
# TAB 3: ARRIVALS & SUPPLY
# ==========================================================================
with tab_supply:
    left, right = st.columns(2)
    with left:
        section_head("Arrivals over time", "by crop, monthly total")
        arr_trend = f_arr.groupby(["ym", "crop_name"])["arrival_quantity_qtl"].sum().reset_index()
        arr_trend["ym_ts"] = arr_trend["ym"].dt.to_timestamp()
        fig = px.area(arr_trend, x="ym_ts", y="arrival_quantity_qtl", color="crop_name",
                       labels={"ym_ts": "", "arrival_quantity_qtl": "Arrivals (Qtl)"},
                       color_discrete_map=CROP_COLORS)
        fig.update_layout(**CHART_TEMPLATE, height=400)
        show_chart(fig, width='stretch')
    with right:
        section_head("Top mandis by volume", "")
        top_mandis = f_arr.groupby("mandi_id")["arrival_quantity_qtl"].sum().sort_values(ascending=False).head(10).reset_index()
        top_mandis = top_mandis.merge(master[["mandi_id", "mandi_name"]], on="mandi_id", how="left")
        fig2 = px.bar(top_mandis.sort_values("arrival_quantity_qtl"), x="arrival_quantity_qtl", y="mandi_name",
                       orientation="h", labels={"arrival_quantity_qtl": "Arrivals (Qtl)", "mandi_name": ""},
                       color_discrete_sequence=[PRIMARY])
        fig2.update_layout(**CHART_TEMPLATE, height=400)
        show_chart(fig2, width='stretch')

    section_head("Farmer footfall vs. average lot size", "bubble size = total volume")
    fs = f_arr.groupby("mandi_id").agg(farmers=("farmer_count", "sum"), qty=("arrival_quantity_qtl", "sum")).reset_index()
    fs = fs.merge(master[["mandi_id", "mandi_name", "state"]], on="mandi_id", how="left")
    fs["avg_lot_qtl"] = fs["qty"] / fs["farmers"].replace(0, np.nan)
    fig3 = px.scatter(fs, x="farmers", y="avg_lot_qtl", size="qty", color="state", hover_name="mandi_name",
                       color_discrete_sequence=[PRIMARY, "#5B7FA6", GOOD, DANGER, SECONDARY, "#8D93A6"],
                       labels={"farmers": "Farmers served", "avg_lot_qtl": "Avg lot size (Qtl/farmer)"},
                       title="Which mandis serve many small farmers vs. few large sellers?")
    fig3.update_layout(**CHART_TEMPLATE, height=420)
    show_chart(fig3, width='stretch')

# ==========================================================================
# TAB 4: LOGISTICS
# ==========================================================================
with tab_logistics:
    left, right = st.columns(2)
    with left:
        section_head("Transit time distribution", "")
        fig = px.histogram(f_trans, x="clean_transit_hours", nbins=30, color_discrete_sequence=[PRIMARY],
                            labels={"clean_transit_hours": "Transit hours"})
        fig.update_layout(**CHART_TEMPLATE, height=380)
        show_chart(fig, width='stretch')
    with right:
        section_head("Avg delay by warehouse", "")
        wh = f_trans.groupby("destination_warehouse").agg(
            trips=("trip_id", "count"), avg_hrs=("clean_transit_hours", "mean")
        ).reset_index().sort_values("avg_hrs", ascending=False)
        fig2 = px.bar(wh, x="destination_warehouse", y="avg_hrs", color="avg_hrs",
                       color_continuous_scale=RISK_RAMP,
                       labels={"destination_warehouse": "", "avg_hrs": "Avg transit hrs"})
        fig2.update_layout(coloraxis_showscale=False, **CHART_TEMPLATE, height=380)
        show_chart(fig2, width='stretch')

    section_head("Distance vs. transit time", "spot the outlier routes")
    fig3 = px.scatter(f_trans, x="distance_km", y="clean_transit_hours", color="destination_warehouse",
                       trendline="ols" if len(f_trans) > 5 else None,
                       labels={"distance_km": "Distance (km)", "clean_transit_hours": "Transit hours"})
    fig3.update_layout(**CHART_TEMPLATE, height=420)
    show_chart(fig3, width='stretch')

# ==========================================================================
# TAB 5: WEATHER
# ==========================================================================
with tab_weather:
    w_f = weather[weather["clean_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()
    w_f["ym"] = w_f["clean_date"].dt.to_period("M").dt.to_timestamp()
    left, right = st.columns(2)
    with left:
        section_head("Temperature & rainfall", "over time")
        wt = w_f.groupby("ym").agg(avg_temp=("temperature_celsius", "mean"), avg_rain=("rainfall_mm", "mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wt["ym"], y=wt["avg_temp"], name="Avg temp (°C)", line=dict(color=DANGER)))
        fig.add_trace(go.Bar(x=wt["ym"], y=wt["avg_rain"], name="Avg rainfall (mm)", marker_color="#7E9FA3", yaxis="y2", opacity=0.6))
        fig.update_layout(
            **{k: v for k, v in CHART_TEMPLATE.items() if k != "yaxis"}, height=380,
            yaxis=dict(title="°C", gridcolor=GRID, linecolor=LINE),
            yaxis2=dict(title="mm", overlaying="y", side="right", showgrid=False),
        )
        show_chart(fig, width='stretch')
    with right:
        section_head("Rainfall vs. arrivals", "does heavy rain suppress supply?")
        arr_month = f_arr.groupby(f_arr["clean_date"].dt.to_period("M").dt.to_timestamp())["arrival_quantity_qtl"].sum().reset_index()
        arr_month.columns = ["ym", "arrivals"]
        merged_w = wt.merge(arr_month, on="ym", how="inner")
        fig2 = px.scatter(merged_w, x="avg_rain", y="arrivals", trendline="ols" if len(merged_w) > 3 else None,
                           labels={"avg_rain": "Avg monthly rainfall (mm)", "arrivals": "Arrivals (Qtl)"},
                           color_discrete_sequence=[PRIMARY])
        fig2.update_layout(**CHART_TEMPLATE, height=380)
        show_chart(fig2, width='stretch')
    st.caption("Weather sensors aren't tagged to individual mandis in the source data, so this view is network-wide, aligned by month.")

# ==========================================================================
# TAB 6: DATA QUALITY
# ==========================================================================
with tab_quality:
    section_head("What the raw files actually looked like", "")
    st.write(
        "Every source file had entry errors: wrong units, swapped decimals, inconsistent "
        "mandi ID formats, dates in 4+ formats. Rows that couldn't be trusted were "
        "excluded rather than guessed at, so every KPI above is computed on the clean subset."
    )
    q = pd.DataFrame(quality_log).T
    q.columns = ["Raw rows", "Usable rows", "Dropped %"]
    st.dataframe(q, width='stretch')
    st.markdown(
        """
        - **Mandi IDs** arrived as `MANDI_001`, `MANDI-042`, `mandi_019`, `MANDI050`, and bare `'038'`, all normalized to one `MANDI_0NN` format before any join.
        - **Prices** under ₹50/quintal are decimal-entry errors (real modal prices for these six crops run ₹1,000–8,000/qtl), so they are set to missing rather than used.
        - **Transit hours** occasionally captured a 4-digit year (e.g. `2026`) instead of a duration, so anything above 200 hours is dropped.
        - **Arrival quantity** had a small number of negative values, corrected with `abs()`.
        """
    )

    section_head("Filtered data", "for your own analysis")
    st.dataframe(valued[["clean_date", "mandi_id", "crop_name", "arrival_quantity_qtl", "modal_price", "market_value"]].sort_values("clean_date", ascending=False).head(500), width='stretch')
    st.download_button(
        " Download filtered arrivals + valuation as CSV",
        data=valued.to_csv(index=False).encode("utf-8"),
        file_name="mandi_filtered_valuation.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown(
    f"""<div class="app-footer">
        <span>MandiPulse. Built from clean_mandi_master, clean_price_and_msp, clean_mandi_arrivals,
        clean_transport_logistics &amp; clean_weather_sensors</span>
        <span>Showing {len(sel_mandis)} of {master['mandi_id'].nunique()} mandis · {start_date:%d %b %Y}–{end_date:%d %b %Y}</span>
       </div>""",
    unsafe_allow_html=True,
)