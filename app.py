import os
import time
import json
import textwrap
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

# Optional (used for evidence panel regression)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Optional (AI Assistant)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Smoking Intelligence Platform"
DATA_PATH = os.getenv("DATA_PATH", "data/italy_smoking_master_mapready.csv")
GEOJSON_PATH = os.getenv("GEOJSON_PATH", "assets/italy_regions.geojson")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # you set this in Railway Variables

EXPOSE_PORT = int(os.getenv("PORT", "8080"))

PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
}


# -----------------------------
# Helpers
# -----------------------------
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}%"


def fmt_pp(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f} pp"


def wrap_title(s, width=40):
    if not s:
        return ""
    return "<br>".join(textwrap.wrap(s, width=width))


def compute_rank_latest(df, value_col="prevalence"):
    # rank regions at latest available year in the filtered subset (by sex/age filters etc.)
    if df.empty:
        return {}
    latest_year = int(df["year"].max())
    d = df[df["year"] == latest_year].copy()
    if d.empty:
        return {}
    d = d.groupby("region", as_index=False)[value_col].mean()
    d["rank"] = d[value_col].rank(ascending=False, method="min").astype(int)
    return dict(zip(d["region"], d["rank"])), latest_year


def filter_df(df, region=None, region_a=None, region_b=None, year_min=None, year_max=None, sex=None, age_group=None):
    d = df.copy()

    if year_min is not None:
        d = d[d["year"] >= int(year_min)]
    if year_max is not None:
        d = d[d["year"] <= int(year_max)]

    if sex and sex != "All" and "sex" in d.columns:
        d = d[d["sex"] == sex]

    if age_group and age_group != "All" and "age_group" in d.columns:
        d = d[d["age_group"] == age_group]

    # single region mode
    if region and region != "All":
        d = d[d["region"] == region]

    # comparison mode
    if region_a and region_a != "All":
        pass
    if region_b and region_b != "All":
        pass

    return d


def make_kpi_card(title, value, subtitle):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="kpi-title"),
                html.Div(value, className="kpi-value"),
                html.Div(subtitle, className="kpi-subtitle"),
            ]
        ),
        className="kpi-card",
    )


def load_assets():
    df = pd.read_csv(DATA_PATH)

    # normalize expected columns
    df.columns = [c.strip() for c in df.columns]
    if "Year" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"Year": "year"})
    if "Region" in df.columns and "region" not in df.columns:
        df = df.rename(columns={"Region": "region"})

    # ensure required columns exist
    required = ["region", "year", "prevalence"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {DATA_PATH}")

    df["year"] = df["year"].astype(int)
    df["prevalence"] = pd.to_numeric(df["prevalence"], errors="coerce")

    # load geojson
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geo = json.load(f)

    # build region list (prefer data regions)
    regions = sorted(df["region"].dropna().unique().tolist())

    # optional dimensions
    sexes = ["All"]
    if "sex" in df.columns:
        sexes += sorted([x for x in df["sex"].dropna().unique().tolist() if str(x).strip()])

    ages = ["All"]
    if "age_group" in df.columns:
        ages += sorted([x for x in df["age_group"].dropna().unique().tolist() if str(x).strip()])

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())

    return df, geo, regions, sexes, ages, year_min, year_max


DF, ITALY_GEO, REGIONS, SEXES, AGES, YEAR_MIN, YEAR_MAX = load_assets()

DEFAULT_A = "Lazio" if "Lazio" in REGIONS else REGIONS[0]
DEFAULT_B = "Lombardia" if "Lombardia" in REGIONS else (REGIONS[1] if len(REGIONS) > 1 else REGIONS[0])


def build_map(df_for_map, map_year):
    d = df_for_map[df_for_map["year"] == int(map_year)].copy()
    if d.empty:
        fig = px.choropleth(title="No data for selected year")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        return fig

    d = d.groupby("region", as_index=False)["prevalence"].mean()

    # Try common geojson keys
    candidate_keys = []
    feat0 = ITALY_GEO.get("features", [{}])[0]
    props = (feat0.get("properties") or {})
    for k in ["reg_name", "name", "NAME_1", "region", "Regione"]:
        if k in props:
            candidate_keys.append(k)
    if not candidate_keys:
        # fall back: guess the first property key
        candidate_keys = list(props.keys())[:1] if props else ["name"]

    feature_key = f"properties.{candidate_keys[0]}"

    fig = px.choropleth(
        d,
        geojson=ITALY_GEO,
        locations="region",
        featureidkey=feature_key,
        color="prevalence",
        color_continuous_scale="Plasma",
        title=f"Smoking prevalence map ({int(map_year)})",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        title=dict(x=0.02, y=0.98),
    )
    return fig


def build_trend_compare(df_filtered, region_a, region_b, year_min, year_max):
    if df_filtered.empty:
        fig = px.line(title="No data for selected filters")
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        return fig

    d = df_filtered.copy()
    d = d[(d["region"].isin([region_a, region_b]))].copy()
    if d.empty:
        fig = px.line(title="No data for Region A/B under current filters")
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        return fig

    d = d.groupby(["year", "region"], as_index=False)["prevalence"].mean()
    title = wrap_title(f"Smoking prevalence trend (Region A vs Region B)", 45)

    fig = px.line(
        d,
        x="year",
        y="prevalence",
        color="region",
        markers=True,
        title=title,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=80, b=10),
        legend_title_text="Region",
        title=dict(x=0.02),
    )
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(title="Prevalence (%)")
    return fig


def compute_kpis_for_region(df_filtered, region):
    d = df_filtered[df_filtered["region"] == region].copy()
    if d.empty:
        return dict(latest=np.nan, latest_year=np.nan, change=np.nan, change_span="", peak=np.nan, peak_year=np.nan)

    d = d.groupby("year", as_index=False)["prevalence"].mean().sort_values("year")
    latest_year = int(d["year"].max())
    latest_val = float(d[d["year"] == latest_year]["prevalence"].iloc[0])

    # change uses first year in selected range
    first_year = int(d["year"].min())
    first_val = float(d[d["year"] == first_year]["prevalence"].iloc[0])
    change = latest_val - first_val

    # peak
    idx = d["prevalence"].idxmax()
    peak_val = float(d.loc[idx, "prevalence"])
    peak_year = int(d.loc[idx, "year"])

    return dict(
        latest=latest_val,
        latest_year=latest_year,
        change=change,
        change_span=f"{first_year} to {latest_year}",
        peak=peak_val,
        peak_year=peak_year,
    )


def evidence_panel_data(df_scope):
    # Evidence based on the currently filtered dataset (not just A/B),
    # using the latest year available within df_scope.
    if df_scope.empty:
        return None

    latest_year = int(df_scope["year"].max())
    d = df_scope[df_scope["year"] == latest_year].copy()
    if d.empty:
        return None

    # correlation candidates
    feature_candidates = []
    for c in ["unemployment_rate", "policy_index", "sunshine_hours"]:
        if c in d.columns:
            feature_candidates.append(c)

    # Correlations (region-level)
    corr_rows = []
    d_region = d.groupby("region", as_index=False)[["prevalence"] + feature_candidates].mean(numeric_only=True)

    if len(d_region) >= 3 and feature_candidates:
        corr = d_region[["prevalence"] + feature_candidates].corr(numeric_only=True)["prevalence"].drop("prevalence")
        for feat, val in corr.items():
            corr_rows.append((feat, safe_float(val)))

    # Regression (simple linear regression)
    reg_result = None
    if len(d_region) >= 5 and feature_candidates:
        X = d_region[feature_candidates].fillna(d_region[feature_candidates].median(numeric_only=True))
        y = d_region["prevalence"].values
        model = LinearRegression()
        model.fit(X, y)
        yhat = model.predict(X)
        r2 = r2_score(y, yhat)
        coefs = list(zip(feature_candidates, model.coef_.tolist()))
        # strongest driver by absolute coefficient
        strongest = max(coefs, key=lambda t: abs(t[1])) if coefs else (None, None)
        reg_result = {
            "latest_year": latest_year,
            "r2": float(r2),
            "coefs": coefs,
            "strongest_driver": strongest[0],
            "strongest_coef": float(strongest[1]) if strongest[1] is not None else None,
        }

    return {
        "latest_year": latest_year,
        "corr_rows": corr_rows,
        "reg": reg_result,
        "features": feature_candidates,
    }


# -----------------------------
# OpenAI: safe call with retries (429 handling)
# -----------------------------
def call_openai_chat(system_msg, user_msg, max_tokens=350, temperature=0.25):
    """
    Uses OPENAI_API_KEY + OPENAI_MODEL from environment.
    Retries on 429 rate limit with short backoff.
    Returns (ok: bool, text: str)
    """
    if not OPENAI_API_KEY or not OpenAI:
        return False, "AI Assistant is not configured. Add OPENAI_API_KEY in Railway Variables to enable it."

    client = OpenAI(api_key=OPENAI_API_KEY)

    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.choices[0].message.content.strip()
            return True, text
        except Exception as e:
            last_err = str(e)
            # Rate limit / throttling
            if "429" in last_err or "rate" in last_err.lower():
                time.sleep(2 * (attempt + 1))
                continue
            break

    return False, f"AI request failed. {last_err}"


# -----------------------------
# App + Styling
# -----------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title=APP_TITLE,
    suppress_callback_exceptions=True,
)
server = app.server

GLOBAL_CSS = """
:root{
  --card-radius: 16px;
  --card-border: rgba(0,0,0,.12);
  --shadow: 0 6px 18px rgba(0,0,0,.08);
}
body{ background:#fff; }
.container-max{ max-width: 1320px; margin: 0 auto; padding: 26px 18px 40px; }

.h-title{ font-size: 34px; letter-spacing: .12em; font-weight: 700; margin: 0; }
.h-sub{ color: rgba(0,0,0,.55); margin-top: 8px; }

.panel-card{
  border: 1px solid var(--card-border);
  border-radius: var(--card-radius);
  box-shadow: var(--shadow);
}

.kpi-card{
  border: 1px solid var(--card-border);
  border-radius: var(--card-radius);
  box-shadow: var(--shadow);
}
.kpi-title{ color: rgba(0,0,0,.6); font-size: 15px; }
.kpi-value{ font-size: 34px; font-weight: 700; line-height: 1.08; margin-top: 6px; }
.kpi-subtitle{ color: rgba(0,0,0,.55); margin-top: 6px; }

.filters-title{ font-size: 18px; font-weight: 700; margin-bottom: 10px; }
.filter-label{ color: rgba(0,0,0,.65); margin-top: 10px; margin-bottom: 6px; }

.graph-card{
  border: 1px solid var(--card-border);
  border-radius: var(--card-radius);
  box-shadow: var(--shadow);
  padding: 14px 14px 10px;
}

.graph-title{
  font-size: 22px;
  font-weight: 700;
  margin: 6px 0 10px 4px;
}

.small-note{ color: rgba(0,0,0,.55); font-size: 13px; margin-top: 8px; }

.ai-card{
  border: 1px solid var(--card-border);
  border-radius: var(--card-radius);
  box-shadow: var(--shadow);
  padding: 16px;
}

.ai-title{ font-size: 28px; font-weight: 800; margin: 0; }
.ai-sub{ color: rgba(0,0,0,.65); margin-top: 8px; }

.modebar{ transform: translateY(10px); } /* pushes the plotly toolbar down slightly */
.js-plotly-plot .plotly .modebar{ top: 10px !important; } /* avoid overlapping titles */
"""

app.layout = html.Div(
    [
        html.Style(GLOBAL_CSS),
        html.Div(
            className="container-max",
            children=[
                html.H1(APP_TITLE.upper(), className="h-title"),
                html.Div("Regional trends, comparative analysis, and forecasting", className="h-sub"),
                html.Div(style={"height": "18px"}),

                dbc.Row(
                    [
                        # Sidebar
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div("Filters", className="filters-title"),

                                        html.Div("Region A", className="filter-label"),
                                        dcc.Dropdown(
                                            id="region_a",
                                            options=[{"label": r, "value": r} for r in REGIONS],
                                            value=DEFAULT_A,
                                            clearable=False,
                                        ),

                                        html.Div("Region B", className="filter-label"),
                                        dcc.Dropdown(
                                            id="region_b",
                                            options=[{"label": r, "value": r} for r in REGIONS],
                                            value=DEFAULT_B,
                                            clearable=False,
                                        ),

                                        html.Div("Year range", className="filter-label"),
                                        dcc.RangeSlider(
                                            id="year_range",
                                            min=YEAR_MIN,
                                            max=YEAR_MAX,
                                            step=1,
                                            value=[max(YEAR_MIN, YEAR_MAX - 14), YEAR_MAX],
                                            marks={YEAR_MIN: str(YEAR_MIN), YEAR_MAX: str(YEAR_MAX)},
                                            allowCross=False,
                                        ),

                                        html.Div("Sex", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sex",
                                            options=[{"label": s, "value": s} for s in SEXES],
                                            value="All",
                                            clearable=False,
                                        ),

                                        html.Div("Age group", className="filter-label"),
                                        dcc.Dropdown(
                                            id="age_group",
                                            options=[{"label": a, "value": a} for a in AGES],
                                            value="All",
                                            clearable=False,
                                        ),

                                        html.Div("Map year", className="filter-label"),
                                        dcc.Slider(
                                            id="map_year",
                                            min=YEAR_MIN,
                                            max=YEAR_MAX,
                                            step=1,
                                            value=YEAR_MAX,
                                            marks={YEAR_MIN: str(YEAR_MIN), YEAR_MAX: str(YEAR_MAX)},
                                        ),

                                        html.Div(className="small-note", children="Tip: Region A vs Region B compares trends using the same filters."),
                                    ]
                                ),
                                className="panel-card",
                            ),
                            width=3,
                        ),

                        # Main
                        dbc.Col(
                            [
                                # KPI row
                                dbc.Row(
                                    [
                                        dbc.Col(html.Div(id="kpi_a_latest"), width=2),
                                        dbc.Col(html.Div(id="kpi_b_latest"), width=2),
                                        dbc.Col(html.Div(id="kpi_diff"), width=2),
                                        dbc.Col(html.Div(id="kpi_a_change"), width=2),
                                        dbc.Col(html.Div(id="kpi_b_change"), width=2),
                                        dbc.Col(html.Div(id="kpi_rank"), width=2),
                                    ],
                                    className="g-3",
                                ),
                                html.Div(style={"height": "14px"}),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    html.Div("Smoking prevalence trend (A vs B)", className="graph-title"),
                                                    dcc.Graph(id="trend_graph", config=PLOT_CONFIG, style={"height": "460px"}),
                                                ],
                                                className="graph-card",
                                            ),
                                            width=7,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    html.Div(id="map_title", className="graph-title"),
                                                    dcc.Graph(id="map_graph", config=PLOT_CONFIG, style={"height": "460px"}),
                                                ],
                                                className="graph-card",
                                            ),
                                            width=5,
                                        ),
                                    ],
                                    className="g-3",
                                ),

                                html.Div(style={"height": "16px"}),

                                # Evidence Panel
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div("Evidence Panel", className="graph-title"),
                                            dbc.Accordion(
                                                [
                                                    dbc.AccordionItem(
                                                        html.Div(id="evidence_corr"),
                                                        title="Correlations",
                                                    ),
                                                    dbc.AccordionItem(
                                                        html.Div(id="evidence_reg"),
                                                        title="Regression Evidence",
                                                    ),
                                                ],
                                                start_collapsed=True,
                                                flush=True,
                                            ),
                                            html.Div(className="small-note", children="Evidence updates using your current filters (sex/age/year range)."),
                                        ]
                                    ),
                                    className="panel-card",
                                ),

                                html.Div(style={"height": "16px"}),

                                # AI Assistant
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div("AI Assistant", className="ai-title"),
                                            html.Div(
                                                "Ask a question and get an explanation grounded in your current filters. "
                                                'Example: "Why is Region A higher than Region B?" or "What should I check next?"',
                                                className="ai-sub",
                                            ),
                                            html.Div(style={"height": "10px"}),

                                            dbc.Alert(
                                                id="ai_status",
                                                children="Note: The assistant activates when OPENAI_API_KEY is set in Railway Variables.",
                                                color="secondary",
                                                className="mb-3",
                                            ),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Input(
                                                            id="ai_input",
                                                            type="text",
                                                            placeholder="Type your question...",
                                                            style={"width": "100%", "height": "44px", "padding": "10px", "borderRadius": "10px"},
                                                        ),
                                                        width=9,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button("Send", id="ai_send", color="primary", style={"width": "100%", "height": "44px"}),
                                                        width=3,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),

                                            html.Div(style={"height": "12px"}),
                                            dbc.Card(
                                                dbc.CardBody(html.Div(id="ai_output", children="")),
                                                style={"borderRadius": "14px", "border": "1px solid rgba(0,0,0,.12)"},
                                            ),
                                        ]
                                    ),
                                    className="ai-card",
                                ),

                                html.Div(style={"height": "10px"}),
                                html.Div("Built with Dash. Deployed on Railway.", className="small-note"),
                            ],
                            width=9,
                        ),
                    ],
                    className="g-3",
                ),
            ],
        ),
    ]
)


# -----------------------------
# Main dashboard callback
# -----------------------------
@callback(
    Output("kpi_a_latest", "children"),
    Output("kpi_b_latest", "children"),
    Output("kpi_diff", "children"),
    Output("kpi_a_change", "children"),
    Output("kpi_b_change", "children"),
    Output("kpi_rank", "children"),
    Output("trend_graph", "figure"),
    Output("map_graph", "figure"),
    Output("map_title", "children"),
    Output("evidence_corr", "children"),
    Output("evidence_reg", "children"),
    Input("region_a", "value"),
    Input("region_b", "value"),
    Input("year_range", "value"),
    Input("sex", "value"),
    Input("age_group", "value"),
    Input("map_year", "value"),
)
def update_dashboard(region_a, region_b, year_range, sex, age_group, map_year):
    y0, y1 = int(year_range[0]), int(year_range[1])

    df_filtered = filter_df(
        DF,
        year_min=y0,
        year_max=y1,
        sex=sex,
        age_group=age_group,
    )

    # KPIs per region
    kpi_a = compute_kpis_for_region(df_filtered, region_a)
    kpi_b = compute_kpis_for_region(df_filtered, region_b)

    # latest diff at each region's latest within filter
    latest_year = int(max(safe_float(kpi_a["latest_year"], -1), safe_float(kpi_b["latest_year"], -1)))
    diff_latest = (kpi_a["latest"] - kpi_b["latest"]) if np.isfinite(kpi_a["latest"]) and np.isfinite(kpi_b["latest"]) else np.nan

    # Rank at latest available year within filtered dataset (region-level)
    rank_map, rank_year = compute_rank_latest(df_filtered)
    ra = rank_map.get(region_a, None)
    rb = rank_map.get(region_b, None)

    # Build KPI cards
    kpi_a_latest = make_kpi_card(
        f"Latest A ({region_a})",
        fmt_pct(kpi_a["latest"]),
        f"Year {kpi_a['latest_year']} • Peak {fmt_pct(kpi_a['peak'])} ({kpi_a['peak_year']})",
    )
    kpi_b_latest = make_kpi_card(
        f"Latest B ({region_b})",
        fmt_pct(kpi_b["latest"]),
        f"Year {kpi_b['latest_year']} • Peak {fmt_pct(kpi_b['peak'])} ({kpi_b['peak_year']})",
    )
    kpi_diff = make_kpi_card(
        "Difference (A - B)",
        fmt_pp(diff_latest),
        "Compared at latest available year",
    )
    kpi_a_change = make_kpi_card(
        f"Change A ({region_a})",
        fmt_pp(kpi_a["change"]),
        kpi_a["change_span"],
    )
    kpi_b_change = make_kpi_card(
        f"Change B ({region_b})",
        fmt_pp(kpi_b["change"]),
        kpi_b["change_span"],
    )
    kpi_rank = make_kpi_card(
        "National rank (latest)",
        f"A #{ra} | B #{rb}" if (ra is not None and rb is not None) else "—",
        f"Year {rank_year}" if "rank_year" in locals() and rank_year else f"Year {y1}",
    )

    # Figures
    fig_trend = build_trend_compare(df_filtered, region_a, region_b, y0, y1)
    fig_map = build_map(df_filtered, map_year)
    map_title = f"Smoking prevalence map ({int(map_year)})"

    # Evidence panel
    ev = evidence_panel_data(df_filtered)
    if not ev:
        corr_block = dbc.Alert("No evidence available for the current filters.", color="secondary")
        reg_block = dbc.Alert("No regression evidence available for the current filters.", color="secondary")
    else:
        # Correlations
        if ev["corr_rows"]:
            corr_table = dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Feature"), html.Th("Correlation with prevalence")])),
                    html.Tbody(
                        [
                            html.Tr([html.Td(feat), html.Td(f"{val:.3f}")])
                            for feat, val in sorted(ev["corr_rows"], key=lambda t: abs(t[1]), reverse=True)
                        ]
                    ),
                ],
                bordered=False,
                hover=True,
                responsive=True,
                size="sm",
            )
            corr_block = html.Div(
                [
                    html.Div(f"Latest year used: {ev['latest_year']}", className="small-note"),
                    corr_table,
                ]
            )
        else:
            corr_block = dbc.Alert("Correlation features not found (need columns like unemployment_rate, policy_index, sunshine_hours).", color="secondary")

        # Regression
        if ev["reg"]:
            reg = ev["reg"]
            coef_table = dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Feature"), html.Th("Coefficient")])),
                    html.Tbody([html.Tr([html.Td(f), html.Td(f"{c:.6f}")]) for f, c in reg["coefs"]]),
                ],
                bordered=False,
                hover=True,
                responsive=True,
                size="sm",
            )
            reg_block = html.Div(
                [
                    html.Div(f"Latest year used: {reg['latest_year']}", className="small-note"),
                    html.Div(f"Model R² = {reg['r2']:.3f}", style={"fontWeight": 700, "marginTop": "6px"}),
                    html.Div(
                        f"Strongest structural driver: {reg['strongest_driver']} (coef {reg['strongest_coef']:.3f})"
                        if reg["strongest_driver"]
                        else "Strongest structural driver: —",
                        className="small-note",
                    ),
                    html.Div(style={"height": "6px"}),
                    coef_table,
                ]
            )
        else:
            reg_block = dbc.Alert("Not enough data or missing driver columns for regression.", color="secondary")

    return (
        kpi_a_latest,
        kpi_b_latest,
        kpi_diff,
        kpi_a_change,
        kpi_b_change,
        kpi_rank,
        fig_trend,
        fig_map,
        map_title,
        corr_block,
        reg_block,
    )


# -----------------------------
# AI Assistant callback (button-only trigger)
# -----------------------------
@callback(
    Output("ai_output", "children"),
    Output("ai_status", "children"),
    Output("ai_status", "color"),
    Input("ai_send", "n_clicks"),
    State("ai_input", "value"),
    State("region_a", "value"),
    State("region_b", "value"),
    State("year_range", "value"),
    State("sex", "value"),
    State("age_group", "value"),
    prevent_initial_call=True,
)
def ai_assistant(n_clicks, question, region_a, region_b, year_range, sex, age_group):
    if not n_clicks:
        return no_update, no_update, no_update

    if not question or not str(question).strip():
        return "Type a question first.", "Ready.", "secondary"

    # quick config check
    if not OPENAI_API_KEY or not OpenAI:
        return (
            "AI Assistant is not configured. Add OPENAI_API_KEY in Railway Variables to enable it.",
            "AI is not configured. Set OPENAI_API_KEY in Railway Variables.",
            "secondary",
        )

    y0, y1 = int(year_range[0]), int(year_range[1])
    df_filtered = filter_df(DF, year_min=y0, year_max=y1, sex=sex, age_group=age_group)

    kpi_a = compute_kpis_for_region(df_filtered, region_a)
    kpi_b = compute_kpis_for_region(df_filtered, region_b)
    rank_map, rank_year = compute_rank_latest(df_filtered)
    ra = rank_map.get(region_a, None)
    rb = rank_map.get(region_b, None)

    ev = evidence_panel_data(df_filtered)
    reg_summary = ""
    if ev and ev.get("reg"):
        reg = ev["reg"]
        reg_summary = f"Regression R2={reg['r2']:.3f}; strongest_driver={reg['strongest_driver']} (coef={reg['strongest_coef']:.3f})."

    # grounded context
    context = f"""
Filters:
- Region A: {region_a}
- Region B: {region_b}
- Year range: {y0} to {y1}
- Sex: {sex}
- Age group: {age_group}

Current KPIs:
- A latest: {fmt_pct(kpi_a['latest'])} (year {kpi_a['latest_year']}), change: {fmt_pp(kpi_a['change'])} ({kpi_a['change_span']}), peak: {fmt_pct(kpi_a['peak'])} ({kpi_a['peak_year']}), rank: {ra}
- B latest: {fmt_pct(kpi_b['latest'])} (year {kpi_b['latest_year']}), change: {fmt_pp(kpi_b['change'])} ({kpi_b['change_span']}), peak: {fmt_pct(kpi_b['peak'])} ({kpi_b['peak_year']}), rank: {rb}

Evidence:
{reg_summary if reg_summary else "No regression evidence available under current filters."}
""".strip()

    system_msg = (
        "You are a careful data analyst. "
        "Answer using ONLY the provided context, and be explicit when something is not available. "
        "Keep it concise but insightful: 5-10 sentences, plus 2-4 bullet recommendations."
    )
    user_msg = f"{context}\n\nUser question:\n{question.strip()}\n"

    ok, text = call_openai_chat(system_msg, user_msg)
    if ok:
        return text, f"AI ready (model: {OPENAI_MODEL}).", "success"
    return text, "AI request failed. If you see 429, wait 10–30 seconds and retry.", "warning"


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=EXPOSE_PORT, debug=False)
