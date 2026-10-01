# Mandi Market Intelligence Dashboard

A Streamlit dashboard built from five raw agri-market CSVs (price & MSP,
arrivals, transport logistics, weather sensors, mandi master), answering one
business question: **where is the mandi network losing value for farmers,
and why?**

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The `data/` folder already contains the five source CSVs — no upload step
needed. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## What's inside

- **Sidebar filters:** date range, state, district, mandi type, crop, and
  individual mandi — every chart and KPI reacts to these live. Each dropdown
  has a one-click **ALL** option, plus sliders to re-weight the composite risk
  score and a reset button.
- **KPI row:** Estimated market revenue, total arrivals, MSP compliance
  rate, farmer "churn" risk (% of sales below MSP), average transit time,
  and a weather risk index — each with a period-over-period delta.
- **Overview & Risk tab:** the core narrative — a composite, user-tunable
  Mandi Risk Score ranking the riskiest mandis, a radar-chart drill-down for
  any single mandi vs. the network average, a risk leaderboard, and a
  short-term revenue trend projection.
- **🤖 AI Agent tab:** see below.
- **Price & MSP / Arrivals & Supply / Logistics / Weather tabs:** the
  supporting evidence behind the headline story.
- **Data Quality tab:** full transparency on what was wrong with the raw
  files and how each issue was handled, plus a CSV export of the cleaned,
  filtered dataset.

## The AI Agent

Lives in `agent.py` and works on whatever the sidebar filters currently show.

**1. Proactive findings (no API key, no network).** The agent runs ten analyst
checks over the filtered data and returns ranked findings — MSP breach
hotspots, the most MSP-exposed crop, revenue momentum, logistics bottlenecks,
routes that are slow for their distance, volume concentration, price
instability, heat exposure, price-coverage gaps, and seasonal peaks. Each
finding carries a severity, the evidence it used, and a recommended action.
Checks are individually sandboxed, so one failing check can't break the tab.

**2. Ask the data.** Natural-language questions are routed to intents and
answered by *computing over the dataframes* — so every number quoted is real
rather than generated. Ask about risk, MSP, revenue, arrivals, prices,
farmers, logistics, or weather; many answers come with a supporting table.

**3. Online layer: Claude API escalation.** If a server-side Anthropic API key
is configured (see below), questions the rule-based router can't map are
escalated to the Claude API automatically — no visitor ever needs their own
key. Each escalation sends a compact statistical briefing of the current
selection plus the last few turns of that chat session, so follow-up
questions ("what about just wheat?") resolve correctly. This is strictly
optional — findings and the rule-based Q&A above work fully offline without
any key at all.

### Enabling the Claude layer

The app reads `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`, default
`claude-sonnet-5`) from Streamlit secrets, falling back to environment
variables if secrets aren't set up. Nothing is ever hard-coded into the repo.

- **Local run:** copy `.streamlit/secrets.toml.example` to
  `.streamlit/secrets.toml` and fill in your key. That file is gitignored.
- **Streamlit Community Cloud:** skip the file — paste the same two lines
  into your app's **Settings → Secrets**.
- **Any other host:** set `ANTHROPIC_API_KEY` as an environment variable.

Without a key configured, the AI Agent tab still works end-to-end — the
sidebar just says so and every question is answered by the offline engine.

Two accuracy details worth knowing, both caught during testing:

- Crop names are matched on word boundaries. Plain substring matching is
  unsafe here because "rice" sits inside "p*rice*", which made
  "average price of wheat" answer about Rice.
- Trend calculations drop a trailing **partial** month. The source data ends
  mid-month, and leaving that stub in made a truncated month look like a
  collapse in volume, flipping the reported trend.

## Data-cleaning assumptions (documented, not hidden)

- Mandi IDs appeared as `MANDI_001`, `MANDI-042`, `mandi_019`, `MANDI050`,
  and bare `'038'` — all normalized to `MANDI_0NN`.
- Prices under ₹50/quintal are decimal-entry errors (real modal prices for
  these six crops run ₹1,000–8,000/qtl) — treated as missing, not used.
- `clean_transit_hours` values above 200 are corrupted (a 4-digit year
  leaked into the field) and are dropped.
- Negative arrival quantities are sign errors, corrected with `abs()`.
- Market value / "revenue" is estimated as `arrival_quantity_qtl × average
  modal price for that mandi+crop+month` (falling back to the crop's
  network-wide monthly average price where a mandi-specific price is
  missing that month) — an estimate, not a ledger figure, and it's labeled
  as such in the UI.
