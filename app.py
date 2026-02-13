import json
from functools import lru_cache

import numpy as np
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go


# -------------------------
# App setup
# -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# -------------------------
# Load data
# -------------------------
DATA_PATH = "data/italy_smoking_master_mapready.csv"
GEO_PATH = "assets/italy_regions.geojson"

df = pd.read_csv(DATA_PATH)

# Normalize column names defensively (in case CSV casing differs)
df.columns = [c.strip() for c in df.columns]

# Expected columns
COL_REGION = "region"
COL_YEAR = "year"
COL_PREV = "prevalence"
COL_SEX = "sex"
COL_AGE = "age_group"

# Optional evidence columns
EVIDENCE_COLS = ["unemployment_rate", "policy_index", "sunshine_hours"]


@lru_cache(maxsize=1)
def load_geojson():
    with open(GEO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


italy_geo = load_geojson()


# -------------------------
# Helpers
# -------------------------
def safe_unique(series):
    vals = series.dropna().unique().tolist()
    vals = sorted(vals)
    return vals


regions = safe_unique(df[COL_REGION])
sexes = ["All"] + safe_unique(df[COL_SEX]) if COL_SEX in df.columns else ["All"]
ages = ["All"] + safe_unique(df[COL_AGE]) if COL_AGE in df.columns else ["All"]

min_year = int(df[COL_YEAR].min())
max_year = int(df[COL_YEAR].max())

default_a = regions[0] if regions else None
default_b = regions[1] if len(regions) > 1 else default_a


def filter_df(dfin, region=None, year_range=None, sex="All", age="All"):
    d = dfin.copy()

    if region and region != "All":
        d = d[d[COL_REGION] == region]

    if year_range:
        y0, y1 = year_range
        d = d[(d[COL_YEAR] >= y0) & (d[COL_YEAR] <= y1)]

    if COL_SEX in d.columns and sex and sex != "All":
        d = d[d[COL_SEX] == sex]

    if COL_AGE in d.columns and age and age != "All":
        d = d[d[COL_AGE] == age]

    return d


def kpi_card(title, value, subtitle=None):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted", style={"fontSize": "14px"}),
                html.Div(value, style={"fontSize": "30px", "fontWeight": "700", "lineHeight": "1.1"}),
                html.Div(subtitle or "", className="text-muted", style={"fontSize": "14px"}),
            ]
        ),
        style={"borderRadius": "14px", "boxShadow": "0 10px 22px rgba(0,0,0,0.06)"},
    )


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}%"


def fmt_pp(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f} pp"


def latest_year_and_value(d):
    if d.empty:
        return None, None
    y = int(d[COL_YEAR].max())
    v = float(d[d[COL_YEAR] == y][COL_PREV].mean())
    return y, v


def peak_year_and_value(d):
    if d.empty:
        return None, None
    by = d.groupby(COL_YEAR, as_index=False)[COL_PREV].mean()
    i = by[COL_PREV].idxmax()
    return int(by.loc[i, COL_YEAR]), float(by.loc[i, COL_PREV])


def change_between(d, y0, y1):
    if d.empty:
        return None
    a = d[d[COL_YEAR] == y0][COL_PREV].mean() if (d[COL_YEAR] == y0).any() else np.nan
    b = d[d[COL_YEAR] == y1][COL_PREV].mean() if (d[COL_YEAR] == y1).any() else np.nan
    if np.isnan(a) or np.isnan(b):
        return None
    return float(b - a)


def national_rank(df_full, year, sex="All", age="All", region=None):
    d = filter_df(df_full, region="All", year_range=(year, year), sex=sex, age=age)
    if d.empty:
        return None
    by = d.groupby(COL_REGION, as_index=False)[COL_PREV].mean().sort_values(COL_PREV, ascending=False)
    by["rank"] = np.arange(1, len(by) + 1)
    if region is None:
        return None
    r = by[by[COL_REGION] == region]
    if r.empty:
        return None
    return int(r["rank"].iloc[0])


def build_trend_figure(d_a, d_b, region_a, region_b):
    fig = go.Figure()

    if not d_a.empty:
        s_a = d_a.groupby(COL_YEAR, as_index=False)[COL_PREV].mean()
        fig.add_trace(
            go.Scatter(
                x=s_a[COL_YEAR],
                y=s_a[COL_PREV],
                mode="lines+markers",
                name=region_a,
            )
        )

    if region_b and region_b != region_a and (not d_b.empty):
        s_b = d_b.groupby(COL_YEAR, as_index=False)[COL_PREV].mean()
        fig.add_trace(
            go.Scatter(
                x=s_b[COL_YEAR],
                y=s_b[COL_PREV],
                mode="lines+markers",
                name=region_b,
            )
        )

    fig.update_layout(
        title=dict(text="Smoking prevalence trend", x=0, xanchor="left", font=dict(size=22)),
        margin=dict(l=60, r=20, t=70, b=55),  # extra top space avoids title/modebar collision
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Year", showgrid=True)
    fig.update_yaxes(title_text="Prevalence", showgrid=True)

    return fig


def build_map_figure(df_full, map_year, sex="All", age="All"):
    d = filter_df(df_full, region="All", year_range=(map_year, map_year), sex=sex, age=age)
    if d.empty:
        fig = px.choropleth()
        fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
        return fig

    by = d.groupby(COL_REGION, as_index=False)[COL_PREV].mean()

    fig = px.choropleth(
        by,
        geojson=italy_geo,
        locations=COL_REGION,
        featureidkey="properties.reg_name",
        color=COL_PREV,
        projection="mercator",
        title="Smoking prevalence map",
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=70, b=10),  # top space to avoid overlap with modebar
        coloraxis_colorbar=dict(title=COL_PREV),
    )
    return fig


def compute_evidence(df_full, region, year_range=None, sex="All", age="All"):
    """
    Evidence = simple correlations + optional linear regression coefficients.
    Defensive: if columns missing or too few rows, returns safe text.
    """
    d = filter_df(df_full, region=region, year_range=year_range, sex=sex, age=age)
    if d.empty:
        return {"status": "empty"}

    # ensure numeric
    for c in [COL_PREV] + EVIDENCE_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    usable_cols = [c for c in EVIDENCE_COLS if c in d.columns and d[c].notna().sum() >= 3]
    if len(usable_cols) == 0:
        return {"status": "no_evidence_cols"}

    # correlation
    corr = d[[COL_PREV] + usable_cols].corr(numeric_only=True)[COL_PREV].sort_values(ascending=False)

    # simple regression (least squares)
    reg_out = None
    d2 = d[[COL_PREV] + usable_cols].dropna()
    if len(d2) >= max(8, len(usable_cols) * 3):
        X = d2[usable_cols].values
        y = d2[COL_PREV].values
        X = np.column_stack([np.ones(len(X)), X])  # intercept

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        coeffs = pd.DataFrame(
            {"Feature": usable_cols, "Coefficient": beta[1:]}
        ).sort_values("Coefficient", key=lambda s: np.abs(s), ascending=False)

        strongest = coeffs.iloc[0].to_dict()

        reg_out = {
            "r2": float(r2) if not np.isnan(r2) else None,
            "coeffs": coeffs,
            "strongest": strongest,
        }

    return {
        "status": "ok",
        "corr": corr,
        "reg": reg_out,
        "n": int(len(d)),
    }


def evidence_panel_layout(evi, region_label):
    if evi["status"] == "empty":
        return dbc.Alert("No data available for the selected filters.", color="secondary")

    if evi["status"] == "no_evidence_cols":
        return dbc.Alert(
            "Evidence columns are missing from the dataset (expected: unemployment_rate, policy_index, sunshine_hours).",
            color="secondary",
        )

    corr = evi["corr"]
    corr_tbl = pd.DataFrame({"Metric": corr.index, "Correlation": corr.values})
    corr_tbl["Correlation"] = corr_tbl["Correlation"].round(3)

    corr_table = dbc.Table.from_dataframe(
        corr_tbl,
        striped=True,
        bordered=True,
        hover=True,
        size="sm",
    )

    items = [
        dbc.AccordionItem(
            [
                html.P(
                    f"Correlation snapshot for {region_label}. Positive means prevalence increases as the metric increases; negative means the opposite.",
                    className="text-muted",
                ),
                corr_table,
            ],
            title="Correlations",
        )
    ]

    if evi["reg"] is not None:
        reg = evi["reg"]
        r2 = reg["r2"]
        strongest = reg["strongest"]

        coeffs = reg["coeffs"].copy()
        coeffs["Coefficient"] = coeffs["Coefficient"].round(4)
        coeff_table = dbc.Table.from_dataframe(coeffs, striped=True, bordered=True, hover=True, size="sm")

        items.append(
            dbc.AccordionItem(
                [
                    html.P(
                        "Linear model summary: coefficients represent the directional relationship after controlling for other included variables.",
                        className="text-muted",
                    ),
                    html.Div(
                        [
                            html.Strong("Model R²: "),
                            html.Span("—" if r2 is None else f"{r2:.3f}"),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(
                        [
                            html.Strong("Strongest driver (by absolute coefficient): "),
                            html.Span(f"{strongest['Feature']} ({strongest['Coefficient']:.4f})"),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    coeff_table,
                ],
                title="Regression Evidence",
            )
        )

    return dbc.Accordion(items, start_collapsed=True, always_open=False)


# -------------------------
# Layout
# -------------------------
app.layout = dbc.Container(
    [
        html.Div(
            [
                html.H1("SMOKING INTELLIGENCE PLATFORM", style={"letterSpacing": "2px", "fontWeight": "800"}),
                html.Div("Regional trends, comparative analysis, and forecasting", className="text-muted"),
            ],
            style={"marginTop": "18px", "marginBottom": "12px"},
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Filters", style={"fontWeight": "700"}),

                                html.Div("Region A", className="text-muted", style={"marginTop": "8px"}),
                                dcc.Dropdown(
                                    id="region_a",
                                    options=[{"label": r, "value": r} for r in regions],
                                    value=default_a,
                                    clearable=False,
                                ),

                                html.Div("Region B", className="text-muted", style={"marginTop": "12px"}),
                                dcc.Dropdown(
                                    id="region_b",
                                    options=[{"label": r, "value": r} for r in regions],
                                    value=default_b,
                                    clearable=False,
                                ),

                                html.Div("Year range", className="text-muted", style={"marginTop": "12px"}),
                                dcc.RangeSlider(
                                    id="year_range",
                                    min=min_year,
                                    max=max_year,
                                    step=1,
                                    value=[min_year, max_year],
                                    marks={min_year: str(min_year), max_year: str(max_year)},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),

                                html.Div("Sex", className="text-muted", style={"marginTop": "14px"}),
                                dcc.Dropdown(
                                    id="sex",
                                    options=[{"label": s, "value": s} for s in sexes],
                                    value="All",
                                    clearable=False,
                                ),

                                html.Div("Age group", className="text-muted", style={"marginTop": "12px"}),
                                dcc.Dropdown(
                                    id="age",
                                    options=[{"label": a, "value": a} for a in ages],
                                    value="All",
                                    clearable=False,
                                ),

                                html.Div("Map year", className="text-muted", style={"marginTop": "14px"}),
                                dcc.Slider(
                                    id="map_year",
                                    min=min_year,
                                    max=max_year,
                                    step=1,
                                    value=max_year,
                                    marks={min_year: str(min_year), max_year: str(max_year)},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ]
                        ),
                        style={"borderRadius": "16px", "boxShadow": "0 10px 22px rgba(0,0,0,0.06)"},
                    ),
                    width=3,
                ),

                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="kpi_latest_a"), width=2),
                                dbc.Col(html.Div(id="kpi_latest_b"), width=2),
                                dbc.Col(html.Div(id="kpi_diff"), width=2),
                                dbc.Col(html.Div(id="kpi_change_a"), width=2),
                                dbc.Col(html.Div(id="kpi_change_b"), width=2),
                                dbc.Col(html.Div(id="kpi_rank"), width=2),
                            ],
                            className="g-2",
                            style={"marginBottom": "10px"},
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            dcc.Graph(
                                                id="trend_fig",
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                    "responsive": True,
                                                },
                                                style={"height": "430px"},
                                            )
                                        ),
                                        style={"borderRadius": "16px", "boxShadow": "0 10px 22px rgba(0,0,0,0.06)"},
                                    ),
                                    width=7,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            dcc.Graph(
                                                id="map_fig",
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                    "responsive": True,
                                                },
                                                style={"height": "430px"},
                                            )
                                        ),
                                        style={"borderRadius": "16px", "boxShadow": "0 10px 22px rgba(0,0,0,0.06)"},
                                    ),
                                    width=5,
                                ),
                            ],
                            className="g-2",
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H4("Evidence Panel", style={"fontWeight": "800", "marginBottom": "6px"}),
                                                html.Div(
                                                    "A transparent, data-backed explanation of what correlates with smoking prevalence under your selected filters.",
                                                    className="text-muted",
                                                    style={"marginBottom": "12px"},
                                                ),
                                                html.Div(id="evidence_panel"),
                                            ]
                                        ),
                                        style={"borderRadius": "16px", "boxShadow": "0 10px 22px rgba(0,0,0,0.06)"},
                                    ),
                                    width=12,
                                    style={"marginTop": "10px"},
                                )
                            ]
                        ),
                    ],
                    width=9,
                ),
            ],
            className="g-3",
        ),

        html.Div(
            "Built with Dash. Deployed on Railway.",
            className="text-muted",
            style={"marginTop": "14px", "marginBottom": "30px"},
        ),
    ],
    fluid=True,
)


# -------------------------
# Callback
# -------------------------
@app.callback(
    Output("trend_fig", "figure"),
    Output("map_fig", "figure"),
    Output("kpi_latest_a", "children"),
    Output("kpi_latest_b", "children"),
    Output("kpi_diff", "children"),
    Output("kpi_change_a", "children"),
    Output("kpi_change_b", "children"),
    Output("kpi_rank", "children"),
    Output("evidence_panel", "children"),
    Input("region_a", "value"),
    Input("region_b", "value"),
    Input("year_range", "value"),
    Input("sex", "value"),
    Input("age", "value"),
    Input("map_year", "value"),
)
def update(region_a, region_b, year_range, sex, age, map_year):
    d_a = filter_df(df, region=region_a, year_range=tuple(year_range), sex=sex, age=age)
    d_b = filter_df(df, region=region_b, year_range=tuple(year_range), sex=sex, age=age)

    # Trend
    fig_trend = build_trend_figure(d_a, d_b, region_a, region_b)

    # Map
    fig_map = build_map_figure(df, map_year, sex=sex, age=age)

    # KPIs
    ya, va = latest_year_and_value(d_a)
    yb, vb = latest_year_and_value(d_b)

    latest_a = kpi_card(
        f"Latest A ({region_a})",
        fmt_pct(va),
        f"Year {ya}" + (f" • Peak {fmt_pct(peak_year_and_value(d_a)[1])}" if not d_a.empty else ""),
    )
    latest_b = kpi_card(
        f"Latest B ({region_b})",
        fmt_pct(vb),
        f"Year {yb}" + (f" • Peak {fmt_pct(peak_year_and_value(d_b)[1])}" if not d_b.empty else ""),
    )

    # Difference at latest comparable year
    latest_year = None
    if ya is not None and yb is not None:
        latest_year = min(ya, yb)

    diff_pp = None
    if latest_year is not None:
        a_val = d_a[d_a[COL_YEAR] == latest_year][COL_PREV].mean()
        b_val = d_b[d_b[COL_YEAR] == latest_year][COL_PREV].mean()
        if not (np.isnan(a_val) or np.isnan(b_val)):
            diff_pp = float(a_val - b_val)

    diff_card = kpi_card("Difference (A - B)", fmt_pp(diff_pp), "Compared at latest available year")

    # Change window (use 2012 baseline if available, else first year in range)
    baseline = 2012 if (df[COL_YEAR] == 2012).any() else int(year_range[0])
    endy = int(year_range[1])

    ch_a = change_between(d_a, baseline, endy)
    ch_b = change_between(d_b, baseline, endy)

    change_a = kpi_card(f"Change A ({region_a})", fmt_pp(ch_a), f"{baseline} to {endy}")
    change_b = kpi_card(f"Change B ({region_b})", fmt_pp(ch_b), f"{baseline} to {endy}")

    # Rank latest year
    if ya is not None:
        ra = national_rank(df, ya, sex=sex, age=age, region=region_a)
        rb = national_rank(df, ya, sex=sex, age=age, region=region_b)
        rank_txt = f"A #{ra} | B #{rb}" if (ra is not None and rb is not None) else "—"
        rank_card = kpi_card("National rank (latest)", rank_txt, f"Year {ya}")
    else:
        rank_card = kpi_card("National rank (latest)", "—", f"Year {max_year}")

    # Evidence panel (use Region A as default evidence focus)
    evi = compute_evidence(df, region_a, year_range=tuple(year_range), sex=sex, age=age)
    evidence_ui = evidence_panel_layout(evi, region_a)

    return (
        fig_trend,
        fig_map,
        latest_a,
        latest_b,
        diff_card,
        change_a,
        change_b,
        rank_card,
        evidence_ui,
    )


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=False)
