import os
import json
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Optional: model stats (won't crash if missing)
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
except Exception:
    LinearRegression = None
    r2_score = None

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/italy_smoking_master_mapready.csv"
GEO_PATH = "assets/italy_regions.geojson"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["year"] = df["year"].astype(int)

with open(GEO_PATH, "r", encoding="utf-8") as f:
    italy_geo = json.load(f)

# IMPORTANT: Must match your GeoJSON property key
FEATURE_KEY = "properties.reg_name"

# -----------------------------
# Dimensions / Defaults
# -----------------------------
regions = sorted(df["region"].dropna().unique())
sexes = ["All"] + sorted(df["sex"].dropna().unique()) if "sex" in df.columns else ["All"]
ages = ["All"] + sorted(df["age_group"].dropna().unique()) if "age_group" in df.columns else ["All"]

year_min, year_max = int(df["year"].min()), int(df["year"].max())

default_a = "Lazio" if "Lazio" in regions else regions[0]
default_b = "Lombardia" if "Lombardia" in regions else regions[min(1, len(regions) - 1)]

# Candidate evidence variables (only those present will be used)
EVIDENCE_CANDIDATES = ["unemployment_rate", "policy_index", "sunshine_hours"]
EVIDENCE_FEATURES = [c for c in EVIDENCE_CANDIDATES if c in df.columns]

# -----------------------------
# App
# -----------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    title="Smoking Intelligence Platform",
)
server = app.server

# -----------------------------
# Plotly global config (modebar behavior)
# -----------------------------
GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "toggleSpikelines",
        "zoomIn2d",
        "zoomOut2d",
    ],
}

# -----------------------------
# Helpers
# -----------------------------
def card_kpi(title, value, subtitle=None):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted", style={"fontSize": "0.85rem"}),
                html.Div(value, style={"fontSize": "1.35rem", "fontWeight": 700}),
                html.Div(subtitle or "", className="text-muted", style={"fontSize": "0.85rem"}) if subtitle else html.Div(),
            ]
        ),
        className="shadow-sm",
        style={"borderRadius": "12px"},
    )

def section_card(title, children):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, style={"fontWeight": 700, "marginBottom": "10px"}),
                children,
            ]
        ),
        className="shadow-sm",
        style={"borderRadius": "12px"},
    )

def apply_common_filters(d: pd.DataFrame, year_range, sex, age):
    d = d[(d["year"] >= year_range[0]) & (d["year"] <= year_range[1])]
    if "sex" in d.columns and sex != "All":
        d = d[d["sex"] == sex]
    if "age_group" in d.columns and age != "All":
        d = d[d["age_group"] == age]
    return d

def region_slice(region_name, year_range, sex, age):
    d = df[df["region"] == region_name].copy()
    d = apply_common_filters(d, year_range, sex, age)
    return d

def map_df(year, sex, age):
    d = df[df["year"] == year].copy()
    if "sex" in d.columns and sex != "All":
        d = d[d["sex"] == sex]
    if "age_group" in d.columns and age != "All":
        d = d[d["age_group"] == age]
    return d.groupby("region", as_index=False)["prevalence"].mean()

def safe_mean(arr):
    if arr is None or len(arr) == 0:
        return np.nan
    return float(np.mean(arr))

def compute_kpis(d_region: pd.DataFrame):
    if d_region.empty:
        return None

    latest_year = int(d_region["year"].max())
    first_year = int(d_region["year"].min())

    latest_val = safe_mean(d_region[d_region["year"] == latest_year]["prevalence"].values)
    first_val = safe_mean(d_region[d_region["year"] == first_year]["prevalence"].values)
    change = latest_val - first_val

    by_year = d_region.groupby("year")["prevalence"].mean()
    peak_year = int(by_year.idxmax())
    peak_val = float(by_year.max())

    return {
        "latest_year": latest_year,
        "latest_val": latest_val,
        "first_year": first_year,
        "first_val": first_val,
        "change": change,
        "peak_year": peak_year,
        "peak_val": peak_val,
    }

def compute_rank(region_name, year, sex, age):
    snap = df[df["year"] == year].copy()
    if "sex" in snap.columns and sex != "All":
        snap = snap[snap["sex"] == sex]
    if "age_group" in snap.columns and age != "All":
        snap = snap[snap["age_group"] == age]

    snap = snap.groupby("region", as_index=False)["prevalence"].mean()
    snap = snap.sort_values("prevalence", ascending=False).reset_index(drop=True)
    snap["rank"] = np.arange(1, len(snap) + 1)

    row = snap[snap["region"] == region_name]
    if row.empty:
        return None
    return int(row["rank"].iloc[0])

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.2f}%"

def fmt_pp(x):
    return "—" if pd.isna(x) else f"{x:+.2f} pp"

def build_evidence_global(year_range, sex, age):
    """
    Evidence computed on the currently filtered dataset across all regions.
    If sklearn not available or features missing, returns safe text.
    """
    d = df.copy()
    d = apply_common_filters(d, year_range, sex, age)
    d = d.groupby(["region", "year"], as_index=False).mean(numeric_only=True)

    evidence = {
        "r2": None,
        "coeff": None,
        "corr": None,
        "strongest": None,
    }

    if len(EVIDENCE_FEATURES) == 0:
        return evidence

    # Correlations
    try:
        corr = d[["prevalence"] + EVIDENCE_FEATURES].corr(numeric_only=True)["prevalence"].sort_values(ascending=False)
        evidence["corr"] = corr
    except Exception:
        evidence["corr"] = None

    # Regression coefficients
    if LinearRegression is None or r2_score is None:
        return evidence

    model_df = d.dropna(subset=["prevalence"] + EVIDENCE_FEATURES).copy()
    if model_df.empty:
        return evidence

    X = model_df[EVIDENCE_FEATURES].values
    y = model_df["prevalence"].values

    try:
        model = LinearRegression()
        model.fit(X, y)
        y_hat = model.predict(X)

        evidence["r2"] = float(r2_score(y, y_hat))
        coeff = pd.DataFrame({"Feature": EVIDENCE_FEATURES, "Coefficient": model.coef_})
        coeff["abs"] = coeff["Coefficient"].abs()
        coeff = coeff.sort_values("abs", ascending=False).drop(columns=["abs"]).reset_index(drop=True)
        evidence["coeff"] = coeff

        if not coeff.empty:
            evidence["strongest"] = (coeff.loc[0, "Feature"], float(coeff.loc[0, "Coefficient"]))
    except Exception:
        pass

    return evidence

def narrative(region_a, region_b, k_a, k_b, rank_a, rank_b, diff, evidence):
    """
    Deterministic narrative (no API).
    """
    strongest = evidence.get("strongest")
    r2 = evidence.get("r2")

    strongest_txt = "Not available"
    if strongest:
        strongest_txt = f"{strongest[0]} (coef {strongest[1]:+.3f})"

    r2_txt = "Not available" if r2 is None else f"{r2:.2f}"

    lines = [
        f"Region A is {region_a} at {k_a['latest_val']:.2f}% in {k_a['latest_year']} (rank #{rank_a if rank_a else '—'}).",
        f"Region B is {region_b} at {k_b['latest_val']:.2f}% in {k_b['latest_year']} (rank #{rank_b if rank_b else '—'}).",
        f"Difference (A - B): {diff:+.2f} percentage points.",
        f"Across the filtered dataset, strongest structural driver: {strongest_txt}.",
        f"Model explanatory strength (R²): {r2_txt}.",
    ]
    return " ".join(lines)

# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container(
    fluid=True,
    style={"padding": "18px 18px"},
    children=[
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H4(
                                "Smoking Intelligence Platform",
                                style={"fontWeight": 800, "letterSpacing": "0.12rem", "marginBottom": "4px"},
                            ),
                            html.Div(
                                "Regional trends, comparative analysis, and forecasting",
                                className="text-muted",
                                style={"fontSize": "0.95rem"},
                            ),
                        ]
                    ),
                    width=12,
                ),
            ],
            style={"marginBottom": "14px"},
        ),

        dbc.Row(
            className="g-3",
            children=[
                # Sidebar
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Filters", style={"fontWeight": 700, "marginBottom": "10px"}),

                                dbc.Label("Region A"),
                                dcc.Dropdown(
                                    id="region_a",
                                    options=[{"label": r, "value": r} for r in regions],
                                    value=default_a,
                                    clearable=False,
                                ),

                                dbc.Label("Region B", style={"marginTop": "10px"}),
                                dcc.Dropdown(
                                    id="region_b",
                                    options=[{"label": r, "value": r} for r in regions],
                                    value=default_b,
                                    clearable=False,
                                ),

                                dbc.Label("Year range", style={"marginTop": "14px"}),
                                dcc.RangeSlider(
                                    id="year-range",
                                    min=year_min,
                                    max=year_max,
                                    value=[max(year_min, year_max - 12), year_max],
                                    marks={year_min: str(year_min), year_max: str(year_max)},
                                    step=1,
                                    allowCross=False,
                                ),

                                dbc.Label("Sex", style={"marginTop": "14px"}),
                                dcc.Dropdown(
                                    id="sex",
                                    options=[{"label": s, "value": s} for s in sexes],
                                    value="All",
                                    clearable=False,
                                ),

                                dbc.Label("Age group", style={"marginTop": "14px"}),
                                dcc.Dropdown(
                                    id="age",
                                    options=[{"label": a, "value": a} for a in ages],
                                    value="All",
                                    clearable=False,
                                ),

                                dbc.Label("Map year", style={"marginTop": "16px"}),
                                dcc.Slider(
                                    id="map-year",
                                    min=year_min,
                                    max=year_max,
                                    value=year_max,
                                    marks={year_min: str(year_min), year_max: str(year_max)},
                                    step=1,
                                ),
                            ]
                        ),
                        className="shadow-sm",
                        style={"borderRadius": "12px"},
                    ),
                    width=3,
                ),

                # Main + Evidence
                dbc.Col(
                    [
                        dbc.Row(
                            className="g-2",
                            style={"marginBottom": "12px"},
                            children=[
                                dbc.Col(html.Div(id="kpi_a"), width=2),
                                dbc.Col(html.Div(id="kpi_b"), width=2),
                                dbc.Col(html.Div(id="kpi_diff"), width=2),
                                dbc.Col(html.Div(id="kpi_change_a"), width=2),
                                dbc.Col(html.Div(id="kpi_change_b"), width=2),
                                dbc.Col(html.Div(id="kpi_rank"), width=2),
                            ],
                        ),

                        dbc.Row(
                            className="g-2",
                            children=[
                                dbc.Col(
                                    dcc.Graph(
                                        id="trend",
                                        config=GRAPH_CONFIG,
                                        style={"height": "420px"},
                                    ),
                                    width=7,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="map",
                                        config=GRAPH_CONFIG,
                                        style={"height": "420px"},
                                    ),
                                    width=5,
                                ),
                            ],
                        ),

                        dbc.Row(
                            className="g-2",
                            style={"marginTop": "10px"},
                            children=[
                                dbc.Col(html.Div(id="evidence_panel"), width=12),
                            ],
                        ),
                    ],
                    width=9,
                ),
            ],
        ),
    ],
)

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    Output("kpi_a", "children"),
    Output("kpi_b", "children"),
    Output("kpi_diff", "children"),
    Output("kpi_change_a", "children"),
    Output("kpi_change_b", "children"),
    Output("kpi_rank", "children"),
    Output("trend", "figure"),
    Output("map", "figure"),
    Output("evidence_panel", "children"),
    Input("region_a", "value"),
    Input("region_b", "value"),
    Input("year-range", "value"),
    Input("sex", "value"),
    Input("age", "value"),
    Input("map-year", "value"),
)
def update(region_a, region_b, year_range, sex, age, map_year):
    d_a = region_slice(region_a, year_range, sex, age)
    d_b = region_slice(region_b, year_range, sex, age)

    k_a = compute_kpis(d_a)
    k_b = compute_kpis(d_b)

    empty_fig = go.Figure().update_layout(
        title="No data available",
        height=420,
        margin=dict(l=30, r=10, t=50, b=30),
    )

    if (k_a is None) or (k_b is None):
        evidence_panel = section_card(
            "Evidence Panel",
            html.Div("No evidence available for the selected filters.", className="text-muted"),
        )
        return (
            card_kpi("Latest A", "—"),
            card_kpi("Latest B", "—"),
            card_kpi("Difference", "—"),
            card_kpi("Change A", "—"),
            card_kpi("Change B", "—"),
            card_kpi("Rank (latest)", "—"),
            empty_fig,
            empty_fig,
            evidence_panel,
        )

    latest_year = max(k_a["latest_year"], k_b["latest_year"])

    latest_val_a = safe_mean(d_a[d_a["year"] == k_a["latest_year"]]["prevalence"].values)
    latest_val_b = safe_mean(d_b[d_b["year"] == k_b["latest_year"]]["prevalence"].values)
    diff = latest_val_a - latest_val_b

    rank_a = compute_rank(region_a, latest_year, sex, age)
    rank_b = compute_rank(region_b, latest_year, sex, age)

    kpi_a = card_kpi(
        f"Latest A ({region_a})",
        fmt_pct(latest_val_a),
        f"Year {k_a['latest_year']} • Peak {k_a['peak_val']:.2f}% ({k_a['peak_year']})",
    )
    kpi_b = card_kpi(
        f"Latest B ({region_b})",
        fmt_pct(latest_val_b),
        f"Year {k_b['latest_year']} • Peak {k_b['peak_val']:.2f}% ({k_b['peak_year']})",
    )
    kpi_diff = card_kpi(
        "Difference (A - B)",
        fmt_pp(diff),
        "Compared at latest available year",
    )
    kpi_change_a = card_kpi(
        f"Change A ({region_a})",
        fmt_pp(k_a["change"]),
        f"{k_a['first_year']} to {k_a['latest_year']}",
    )
    kpi_change_b = card_kpi(
        f"Change B ({region_b})",
        fmt_pp(k_b["change"]),
        f"{k_b['first_year']} to {k_b['latest_year']}",
    )

    rank_text = "—"
    if (rank_a is not None) and (rank_b is not None):
        rank_text = f"A #{rank_a} | B #{rank_b}"
    kpi_rank = card_kpi("National rank (latest)", rank_text, f"Year {latest_year}")

    # Trend figure: shorten title + add top margin (prevents cut)
    trend_a = d_a.groupby("year", as_index=False)["prevalence"].mean()
    trend_b = d_b.groupby("year", as_index=False)["prevalence"].mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=trend_a["year"], y=trend_a["prevalence"], mode="lines+markers", name=region_a))
    fig_trend.add_trace(go.Scatter(x=trend_b["year"], y=trend_b["prevalence"], mode="lines+markers", name=region_b))

    fig_trend.update_layout(
        title="Smoking prevalence trend",
        height=420,
        margin=dict(l=40, r=20, t=70, b=45),
        legend_title_text="Region",
        hovermode="x unified",
    )
    fig_trend.update_xaxes(title_text="Year")
    fig_trend.update_yaxes(title_text="Prevalence")

    # Modebar: move it down a bit so it won't sit on title area
    fig_trend.update_layout(modebar=dict(orientation="h", y=1.12, x=1, xanchor="right", yanchor="top"))

    # Map figure
    md = map_df(map_year, sex, age)
    fig_map = px.choropleth(
        md,
        geojson=italy_geo,
        locations="region",
        featureidkey=FEATURE_KEY,
        color="prevalence",
        title=f"Smoking prevalence map ({map_year})",
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig_map.update_layout(modebar=dict(orientation="h", y=1.12, x=1, xanchor="right", yanchor="top"))

    # Evidence panel (global evidence + narrative)
    evidence = build_evidence_global(year_range, sex, age)

    r2_txt = "Not available" if evidence["r2"] is None else f"{evidence['r2']:.3f}"
    strongest_txt = "Not available"
    if evidence["strongest"] is not None:
        strongest_txt = f"{evidence['strongest'][0]} (coef {evidence['strongest'][1]:+.3f})"

    # Correlations table (top 3 strongest by abs)
    corr_block = html.Div("Not available", className="text-muted")
    if evidence["corr"] is not None:
        corr = evidence["corr"].drop(labels=["prevalence"], errors="ignore")
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(3)
        corr_block = html.Ul(
            [html.Li(f"{idx}: {val:+.3f}") for idx, val in corr.items()],
            style={"marginBottom": 0},
        )

    coeff_block = html.Div("Not available", className="text-muted")
    if evidence["coeff"] is not None and not evidence["coeff"].empty:
        coeff_block = dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Feature"), html.Th("Coef")])),
                html.Tbody(
                    [
                        html.Tr([html.Td(row["Feature"]), html.Td(f"{row['Coefficient']:+.3f}")])
                        for _, row in evidence["coeff"].head(5).iterrows()
                    ]
                ),
            ],
            bordered=False,
            hover=True,
            size="sm",
            style={"marginBottom": 0},
        )

    story = narrative(region_a, region_b, k_a, k_b, rank_a, rank_b, diff, evidence)

    evidence_panel = section_card(
        "Evidence Panel",
        dbc.Row(
            className="g-3",
            children=[
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Model strength (R²)", className="text-muted", style={"fontSize": "0.85rem"}),
                                html.Div(r2_txt, style={"fontWeight": 800, "fontSize": "1.15rem"}),
                                html.Div("Computed on filtered data (all regions).", className="text-muted", style={"fontSize": "0.85rem"}),
                            ]
                        ),
                        className="shadow-sm",
                        style={"borderRadius": "12px"},
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Strongest driver", className="text-muted", style={"fontSize": "0.85rem"}),
                                html.Div(strongest_txt, style={"fontWeight": 800, "fontSize": "1.05rem"}),
                                html.Div("Highest absolute coefficient.", className="text-muted", style={"fontSize": "0.85rem"}),
                            ]
                        ),
                        className="shadow-sm",
                        style={"borderRadius": "12px"},
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Top correlations", className="text-muted", style={"fontSize": "0.85rem"}),
                                corr_block,
                            ]
                        ),
                        className="shadow-sm",
                        style={"borderRadius": "12px"},
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Regression coefficients", className="text-muted", style={"fontSize": "0.85rem"}),
                                coeff_block,
                            ]
                        ),
                        className="shadow-sm",
                        style={"borderRadius": "12px"},
                    ),
                    width=3,
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Narrative summary", className="text-muted", style={"fontSize": "0.85rem"}),
                                html.Div(story, style={"lineHeight": "1.45"}),
                            ]
                        ),
                        className="shadow-sm",
                        style={"borderRadius": "12px"},
                    ),
                    width=12,
                ),
            ],
        ),
    )

    return kpi_a, kpi_b, kpi_diff, kpi_change_a, kpi_change_b, kpi_rank, fig_trend, fig_map, evidence_panel

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
