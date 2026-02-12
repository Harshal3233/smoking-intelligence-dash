import json
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# -----------------------------

# -----------------------------
DATA_PATH = "data/italy_smoking_master_mapready.csv"
GEO_PATH = "assets/italy_regions.geojson"

df = pd.read_csv(DATA_PATH)
df["year"] = df["year"].astype(int)

with open(GEO_PATH, "r", encoding="utf-8") as f:
    italy_geo = json.load(f)


FEATURE_KEY = "properties.reg_name"

regions = sorted(df["region"].unique())
sexes = ["All"] + sorted(df["sex"].unique())
ages = ["All"] + sorted(df["age_group"].unique())
year_min, year_max = int(df["year"].min()), int(df["year"].max())

# -----------------------------

# -----------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],  # modern dark
    title="Smoking Intelligence Platform",
)
server = app.server

# -----------------------------

# -----------------------------
def card_kpi(title, value, subtitle=None, icon="📌"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Div(icon, style={"fontSize": "1.2rem", "marginRight": "10px"}),
                        html.Div(
                            [
                                html.Div(title, className="text-muted", style={"fontSize": "0.85rem"}),
                                html.Div(value, style={"fontSize": "1.35rem", "fontWeight": 800}),
                                html.Div(subtitle or "", className="text-muted", style={"fontSize": "0.85rem"}) if subtitle else html.Div(),
                            ]
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                )
            ]
        ),
        className="shadow-sm",
        style={"borderRadius": "18px", "border": "1px solid rgba(255,255,255,0.08)"},
    )

def filter_df(region, year_range, sex, age):
    d = df.copy()
    if region != "All":
        d = d[d["region"] == region]
    d = d[(d["year"] >= year_range[0]) & (d["year"] <= year_range[1])]
    if sex != "All":
        d = d[d["sex"] == sex]
    if age != "All":
        d = d[d["age_group"] == age]
    return d

def map_df(year, sex, age):
    d = df[df["year"] == year].copy()
    if sex != "All":
        d = d[d["sex"] == sex]
    if age != "All":
        d = d[d["age_group"] == age]
    return d.groupby("region", as_index=False)["prevalence"].mean()

# -----------------------------

# -----------------------------
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Div("Smoking Intelligence Platform", style={"fontSize": "1.35rem", "fontWeight": 900}),
                            html.Div("Regional trends • maps • comparisons • drivers", className="text-muted", style={"fontSize": "0.9rem"}),
                        ]
                    ),
                    width=8,
                ),
                dbc.Col(
                    dbc.Badge("Dash • Fast UI", color="primary", className="p-2", style={"marginTop": "12px"}),
                    width=4,
                    style={"textAlign": "right"},
                ),
            ],
            style={"padding": "14px 10px"},
        ),

        dbc.Row(
            [
                
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Controls", style={"fontWeight": 800, "marginBottom": "10px"}),

                                dbc.Label("Region"),
                                dcc.Dropdown(
                                    id="region",
                                    options=[{"label": "All", "value": "All"}] + [{"label": r, "value": r} for r in regions],
                                    value="All",
                                    clearable=False,
                                ),

                                dbc.Label("Year range", style={"marginTop": "12px"}),
                                dcc.RangeSlider(
                                    id="year-range",
                                    min=year_min,
                                    max=year_max,
                                    value=[max(year_min, year_max - 12), year_max],
                                    marks={year_min: str(year_min), year_max: str(year_max)},
                                    step=1,
                                    allowCross=False,
                                ),

                                dbc.Label("Sex", style={"marginTop": "12px"}),
                                dcc.Dropdown(
                                    id="sex",
                                    options=[{"label": s, "value": s} for s in sexes],
                                    value="All",
                                    clearable=False,
                                ),

                                dbc.Label("Age group", style={"marginTop": "12px"}),
                                dcc.Dropdown(
                                    id="age",
                                    options=[{"label": a, "value": a} for a in ages],
                                    value="All",
                                    clearable=False,
                                ),

                                html.Hr(),

                                dbc.Label("Map year"),
                                dcc.Slider(
                                    id="map-year",
                                    min=year_min,
                                    max=year_max,
                                    value=year_max,
                                    marks={year_min: str(year_min), year_max: str(year_max)},
                                    step=1,
                                ),

                                html.Div(className="text-muted", style={"fontSize": "0.85rem", "marginTop": "10px"},
                                         children="Next: Forecast tab, Evidence panel, Chat assistant, PDF report."),
                            ]
                        ),
                        className="shadow-sm",
                        style={"borderRadius": "18px", "border": "1px solid rgba(255,255,255,0.08)"},
                    ),
                    width=3,
                ),

                
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="kpi1"), width=3),
                                dbc.Col(html.Div(id="kpi2"), width=3),
                                dbc.Col(html.Div(id="kpi3"), width=3),
                                dbc.Col(html.Div(id="kpi4"), width=3),
                            ],
                            style={"marginBottom": "12px"},
                        ),

                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="trend"), width=6),
                                dbc.Col(dcc.Graph(id="map"), width=6),
                            ]
                        ),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    style={"paddingBottom": "20px"},
)

# -----------------------------

# -----------------------------
@app.callback(
    Output("kpi1", "children"),
    Output("kpi2", "children"),
    Output("kpi3", "children"),
    Output("kpi4", "children"),
    Output("trend", "figure"),
    Output("map", "figure"),
    Input("region", "value"),
    Input("year-range", "value"),
    Input("sex", "value"),
    Input("age", "value"),
    Input("map-year", "value"),
)
def update(region, year_range, sex, age, map_year):
    d = filter_df(region, year_range, sex, age)

    if d.empty:
        k1 = card_kpi("Latest prevalence", "—", "No data")
        k2 = card_kpi("Change", "—", "")
        k3 = card_kpi("Peak", "—", "")
        k4 = card_kpi("Rank", "—", "")
        empty_fig = px.line(title="No data")
        empty_fig.update_layout(template="plotly_dark", height=320)
        return k1, k2, k3, k4, empty_fig, empty_fig

    latest_year = int(d["year"].max())
    first_year = int(d["year"].min())

    latest_val = float(d[d["year"] == latest_year]["prevalence"].mean())
    first_val = float(d[d["year"] == first_year]["prevalence"].mean())
    change = latest_val - first_val

    by_year = d.groupby("year")["prevalence"].mean()
    peak_year = int(by_year.idxmax())
    peak_val = float(by_year.max())

    rank = None
    if region != "All":
        snap = df[df["year"] == latest_year].groupby("region")["prevalence"].mean().sort_values(ascending=False)
        rank = int(list(snap.index).index(region) + 1)

    k1 = card_kpi("Latest prevalence", f"{latest_val:.2f}%", f"{latest_year}", "📈")
    k2 = card_kpi("Change", f"{change:+.2f} pp", f"{first_year}→{latest_year}", "🔁")
    k3 = card_kpi("Peak", f"{peak_val:.2f}%", f"{peak_year}", "🏔️")
    k4 = card_kpi("Rank", f"#{rank}" if rank else "—", f"{latest_year}", "🏁")

    trend = d.groupby("year", as_index=False)["prevalence"].mean()
    fig_trend = px.line(trend, x="year", y="prevalence", markers=True, title="Trend")
    fig_trend.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=50, b=10))

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
    fig_map.update_layout(template="plotly_dark", height=430, margin=dict(l=10, r=10, t=50, b=10))

    return k1, k2, k3, k4, fig_trend, fig_map

if __name__ == "__main__":
    app.run_server(debug=True)
