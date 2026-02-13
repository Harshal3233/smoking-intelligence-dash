import os
import json
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

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

# IMPORTANT: This must match your GeoJSON property key
FEATURE_KEY = "properties.reg_name"

# -----------------------------
# Dimensions / Defaults
# -----------------------------
regions = sorted(df["region"].dropna().unique())
sexes = ["All"] + sorted(df["sex"].dropna().unique())
ages = ["All"] + sorted(df["age_group"].dropna().unique())

year_min, year_max = int(df["year"].min()), int(df["year"].max())

default_a = "Lazio" if "Lazio" in regions else regions[0]
default_b = "Lombardia" if "Lombardia" in regions else regions[min(1, len(regions) - 1)]

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

def apply_common_filters(d: pd.DataFrame, year_range, sex, age):
    d = d[(d["year"] >= year_range[0]) & (d["year"] <= year_range[1])]
    if sex != "All":
        d = d[d["sex"] == sex]
    if age != "All":
        d = d[d["age_group"] == age]
    return d

def region_slice(region_name, year_range, sex, age):
    d = df[df["region"] == region_name].copy()
    d = apply_common_filters(d, year_range, sex, age)
    return d

def map_df(year, sex, age):
    d = df[df["year"] == year].copy()
    if sex != "All":
        d = d[d["sex"] == sex]
    if age != "All":
        d = d[d["age_group"] == age]
    return d.groupby("region", as_index=False)["prevalence"].mean()

def safe_mean(series):
    if series is None or len(series) == 0:
        return np.nan
    return float(np.mean(series))

def compute_kpis(d_region: pd.DataFrame):
    """
    Returns:
      latest_year, latest_val, first_year, first_val, change, peak_year, peak_val
    """
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
    if sex != "All":
        snap = snap[snap["sex"] == sex]
    if age != "All":
        snap = snap[snap["age_group"] == age]

    snap = snap.groupby("region", as_index=False)["prevalence"].mean()
    snap = snap.sort_values("prevalence", ascending=False).reset_index(drop=True)
    snap["rank"] = np.arange(1, len(snap) + 1)

    row = snap[snap["region"] == region_name]
    if row.empty:
        return None
    return int(row["rank"].iloc[0])

# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H4("Smoking Intelligence Platform", style={"fontWeight": 700, "letterSpacing": "0.08rem"}),
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
            style={"padding": "20px 10px"},
        ),

        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Filters", style={"fontWeight": 600, "marginBottom": "10px"}),

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

                # Main panel
                dbc.Col(
                    [
                        # KPI row (comparison-ready)
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="kpi_a"), width=2),
                                dbc.Col(html.Div(id="kpi_b"), width=2),
                                dbc.Col(html.Div(id="kpi_diff"), width=2),
                                dbc.Col(html.Div(id="kpi_change_a"), width=2),
                                dbc.Col(html.Div(id="kpi_change_b"), width=2),
                                dbc.Col(html.Div(id="kpi_rank"), width=2),
                            ],
                            className="g-2",
                            style={"marginBottom": "15px"},
                        ),

                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="trend"), width=6),
                                dbc.Col(dcc.Graph(id="map"), width=6),
                            ],
                            className="g-2",
                        ),
                    ],
                    width=9,
                ),
            ],
            className="g-2",
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
    Input("region_a", "value"),
    Input("region_b", "value"),
    Input("year-range", "value"),
    Input("sex", "value"),
    Input("age", "value"),
    Input("map-year", "value"),
)
def update(region_a, region_b, year_range, sex, age, map_year):
    # Prepare region slices
    d_a = region_slice(region_a, year_range, sex, age)
    d_b = region_slice(region_b, year_range, sex, age)

    k_a = compute_kpis(d_a)
    k_b = compute_kpis(d_b)

    # Empty safeguards
    if (k_a is None) or (k_b is None):
        empty_fig = px.line(title="No data available")
        return (
            card_kpi("Latest A", "—"),
            card_kpi("Latest B", "—"),
            card_kpi("Difference", "—"),
            card_kpi("Change A", "—"),
            card_kpi("Change B", "—"),
            card_kpi("Rank (latest)", "—"),
            empty_fig,
            empty_fig,
        )

    # Latest comparison (use each region's latest within filtered years; usually same)
    latest_year = max(k_a["latest_year"], k_b["latest_year"])

    # If one region doesn't have data in that latest_year, fallback to its own latest_year
    latest_val_a = safe_mean(d_a[d_a["year"] == k_a["latest_year"]]["prevalence"].values)
    latest_val_b = safe_mean(d_b[d_b["year"] == k_b["latest_year"]]["prevalence"].values)

    diff = latest_val_a - latest_val_b

    # Ranks on latest_year with sex/age filter
    rank_a = compute_rank(region_a, latest_year, sex, age)
    rank_b = compute_rank(region_b, latest_year, sex, age)

    # KPI cards
    kpi_a = card_kpi(
        f"Latest A ({region_a})",
        f"{latest_val_a:.2f}%",
        f"Year {k_a['latest_year']} • Peak {k_a['peak_val']:.2f}% ({k_a['peak_year']})",
    )
    kpi_b = card_kpi(
        f"Latest B ({region_b})",
        f"{latest_val_b:.2f}%",
        f"Year {k_b['latest_year']} • Peak {k_b['peak_val']:.2f}% ({k_b['peak_year']})",
    )
    kpi_diff = card_kpi(
        "Difference (A - B)",
        f"{diff:+.2f} pp",
        f"Compared at latest available year",
    )

    kpi_change_a = card_kpi(
        f"Change A ({region_a})",
        f"{k_a['change']:+.2f} pp",
        f"{k_a['first_year']} to {k_a['latest_year']}",
    )
    kpi_change_b = card_kpi(
        f"Change B ({region_b})",
        f"{k_b['change']:+.2f} pp",
        f"{k_b['first_year']} to {k_b['latest_year']}",
    )

    rank_text = "—"
    if (rank_a is not None) and (rank_b is not None):
        rank_text = f"A #{rank_a} | B #{rank_b}"
    kpi_rank = card_kpi("National rank (latest)", rank_text, f"Year {latest_year}")

    # Trend figure (dual line)
    trend_a = d_a.groupby("year", as_index=False)["prevalence"].mean()
    trend_b = d_b.groupby("year", as_index=False)["prevalence"].mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=trend_a["year"],
            y=trend_a["prevalence"],
            mode="lines+markers",
            name=region_a,
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=trend_b["year"],
            y=trend_b["prevalence"],
            mode="lines+markers",
            name=region_b,
        )
    )
    fig_trend.update_layout(
        title="Smoking prevalence trend (Region A vs Region B)",
        height=360,
        legend_title_text="Region",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig_trend.update_xaxes(title_text="Year")
    fig_trend.update_yaxes(title_text="Prevalence")

    # Map figure (independent of A/B; uses sex/age + map year)
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
    fig_map.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))

    return kpi_a, kpi_b, kpi_diff, kpi_change_a, kpi_change_b, kpi_rank, fig_trend, fig_map

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
