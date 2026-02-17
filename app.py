import os
import json
import time
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "data/italy_smoking_master_mapready.csv")
GEOJSON_PATH = os.getenv("GEOJSON_PATH", "assets/italy_regions.geojson")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

APP_TITLE = "SMOKING INTELLIGENCE PLATFORM"
APP_SUBTITLE = "Regional trends, comparative analysis, and forecasting"


# ------------------------------------------------------------
# Helpers: data + stats
# ------------------------------------------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def safe_read_geojson(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_filters(df: pd.DataFrame, region=None, year_min=None, year_max=None, sex=None, age_group=None) -> pd.DataFrame:
    out = df.copy()

    if region and region != "All":
        out = out[out["region"] == region]

    if year_min is not None:
        out = out[out["year"] >= year_min]
    if year_max is not None:
        out = out[out["year"] <= year_max]

    if sex and sex != "All" and "sex" in out.columns:
        out = out[out["sex"] == sex]

    if age_group and age_group != "All" and "age_group" in out.columns:
        out = out[out["age_group"] == age_group]

    return out


def latest_year(df: pd.DataFrame) -> int:
    return int(df["year"].max())


def region_latest_prevalence(df: pd.DataFrame, region: str, sex="All", age_group="All") -> tuple[float, int]:
    d = apply_filters(df, region=region, sex=sex, age_group=age_group)
    if d.empty:
        return (np.nan, np.nan)
    y = latest_year(d)
    v = float(d[d["year"] == y]["prevalence"].mean())
    return v, y


def change_over_period(df: pd.DataFrame, region: str, y0: int, y1: int, sex="All", age_group="All") -> float:
    d = apply_filters(df, region=region, year_min=y0, year_max=y1, sex=sex, age_group=age_group)
    if d.empty:
        return np.nan
    v0 = d[d["year"] == y0]["prevalence"].mean()
    v1 = d[d["year"] == y1]["prevalence"].mean()
    if pd.isna(v0) or pd.isna(v1):
        return np.nan
    return float(v1 - v0)


def peak_value(df: pd.DataFrame, region: str, sex="All", age_group="All") -> tuple[float, int]:
    d = apply_filters(df, region=region, sex=sex, age_group=age_group)
    if d.empty:
        return (np.nan, np.nan)
    idx = d["prevalence"].idxmax()
    return float(d.loc[idx, "prevalence"]), int(d.loc[idx, "year"])


def national_rank_latest(df: pd.DataFrame, region: str, sex="All", age_group="All") -> tuple[int, int]:
    d = apply_filters(df, sex=sex, age_group=age_group)
    if d.empty:
        return (np.nan, np.nan)

    y = latest_year(d)
    latest = d[d["year"] == y].groupby("region", as_index=False)["prevalence"].mean()
    latest["rank"] = latest["prevalence"].rank(ascending=False, method="min").astype(int)

    row = latest[latest["region"] == region]
    if row.empty:
        return (np.nan, y)
    return int(row["rank"].iloc[0]), y


def corr_table(df: pd.DataFrame, region: str, sex="All", age_group="All") -> pd.DataFrame:
    d = apply_filters(df, region=region, sex=sex, age_group=age_group)
    cols = [c for c in ["prevalence", "unemployment_rate", "policy_index", "sunshine_hours", "year"] if c in d.columns]
    if len(cols) < 2 or d.empty:
        return pd.DataFrame(columns=["feature", "correlation"])
    corr = d[cols].corr(numeric_only=True)["prevalence"].drop("prevalence", errors="ignore")
    out = corr.reset_index()
    out.columns = ["feature", "correlation"]
    out["correlation"] = out["correlation"].astype(float)
    out = out.sort_values("correlation", key=lambda s: s.abs(), ascending=False)
    return out


def linear_regression_numpy(df: pd.DataFrame, region: str, sex="All", age_group="All"):
    """
    Simple OLS regression using numpy (no sklearn dependency).
    Predicts prevalence from available features:
    unemployment_rate, policy_index, sunshine_hours, year (if present).
    Returns dict: R2, coefficients table.
    """
    d = apply_filters(df, region=region, sex=sex, age_group=age_group).dropna()

    feature_candidates = ["unemployment_rate", "policy_index", "sunshine_hours", "year"]
    features = [f for f in feature_candidates if f in d.columns]

    if d.empty or len(features) == 0 or "prevalence" not in d.columns:
        return {"r2": np.nan, "coefficients": pd.DataFrame(columns=["feature", "coefficient"])}

    X = d[features].astype(float).values
    y = d["prevalence"].astype(float).values.reshape(-1, 1)

    # Add intercept
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])

    # OLS: beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_design) @ y

    y_hat = X_design @ beta
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    coef = beta[1:].flatten()
    coef_df = pd.DataFrame({"feature": features, "coefficient": coef.astype(float)})
    coef_df = coef_df.sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

    return {"r2": float(r2), "coefficients": coef_df}


def evidence_summary_text(region: str, r2: float, coef_df: pd.DataFrame) -> str:
    if coef_df.empty or pd.isna(r2):
        return f"No regression evidence available for {region} with the current filters."

    top = coef_df.iloc[0]
    driver = str(top["feature"])
    effect = float(top["coefficient"])

    direction = "increases" if effect > 0 else "decreases"
    return (
        f"For {region}, the strongest structural driver in the linear model is **{driver}** "
        f"(coefficient {effect:+.3f}), meaning higher {driver} {direction} predicted prevalence in this dataset. "
        f"Model fit: **R² = {r2:.2f}**."
    )


# ------------------------------------------------------------
# OpenAI helper (optional) - FIXED for Railway proxy crash
# ------------------------------------------------------------
_LAST_AI_CALL_TS = 0.0  # in-process throttle


def _strip_proxy_env():
    # Prevent OpenAI SDK -> httpx proxy wiring causing:
    # TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
    for k in [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    ]:
        os.environ.pop(k, None)


def ai_answer(question: str, context: str) -> tuple[bool, str]:
    """
    Returns (ok, text). If not configured or errors, returns ok=False with message.
    Uses chat.completions (stable) and strips proxy env vars (Railway fix).
    """
    global _LAST_AI_CALL_TS

    if not OPENAI_API_KEY:
        return False, "AI Assistant is not configured. Add OPENAI_API_KEY in Railway Variables to enable it."

    # Light throttle to reduce accidental spam (and 429s)
    now = time.time()
    if now - _LAST_AI_CALL_TS < 2.0:
        return False, "You're sending requests too fast. Wait 2 seconds and try again."
    _LAST_AI_CALL_TS = now

    try:
        _strip_proxy_env()
        from openai import OpenAI
    except Exception:
        return False, "Missing dependency: openai. Add `openai` to requirements.txt."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        system = (
            "You are a data analyst assistant for a Dash dashboard about smoking prevalence in Italian regions. "
            "Answer using the provided context only. Be concise, factual, and reference the numbers given."
        )
        user = f"Context:\n{context}\n\nUser question:\n{question}"

        # Use a compact response to keep cost + rate-limits down
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,  # you set this to gpt-4o-mini in Railway Variables
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=350,
        )

        text = (resp.choices[0].message.content or "").strip()
        return True, text if text else "No response text returned."

    except Exception as e:
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower() or "insufficient_quota" in msg.lower():
            return False, (
                "AI request failed (429). This usually means rate limit or quota. "
                "Try again in a bit, reduce frequency, or check OpenAI billing/limits."
            )
        return False, f"AI request failed: {type(e).__name__}: {msg}"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = safe_read_csv(DATA_PATH)
df["year"] = df["year"].astype(int)

italy_geo = safe_read_geojson(GEOJSON_PATH)

regions = sorted(df["region"].dropna().unique().tolist())

sex_values = ["All"] + (sorted(df["sex"].dropna().unique().tolist()) if "sex" in df.columns else [])
age_values = ["All"] + (sorted(df["age_group"].dropna().unique().tolist()) if "age_group" in df.columns else [])

min_year = int(df["year"].min())
max_year = int(df["year"].max())


# Defaults
default_a = regions[0] if regions else "All"
default_b = regions[1] if len(regions) > 1 else default_a


# ------------------------------------------------------------
# App + styling
# ------------------------------------------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    # Removes clutter so it doesn't overlap headings
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "autoScale2d", "resetScale2d",
        "toggleSpikelines", "hoverCompareCartesian", "hoverClosestCartesian"
    ],
    "responsive": True,
}

CARD_STYLE = {
    "borderRadius": "14px",
    "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
    "border": "1px solid rgba(0,0,0,0.08)",
}

SECTION_TITLE = {"fontSize": "22px", "fontWeight": 600, "marginBottom": "8px"}
MUTED = {"color": "rgba(0,0,0,0.55)"}


def kpi_card(title, value, subtitle=None):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, style={**MUTED, "fontSize": "14px"}),
                html.Div(value, style={"fontSize": "34px", "fontWeight": 650, "lineHeight": "1.1"}),
                html.Div(subtitle or "", style={**MUTED, "marginTop": "6px"}) if subtitle else html.Div(),
            ]
        ),
        style=CARD_STYLE,
    )


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
app.layout = dbc.Container(
    fluid=True,
    style={"padding": "22px 26px"},
    children=[
        html.Div(
            [
                html.Div(APP_TITLE, style={"letterSpacing": "0.12em", "fontSize": "30px", "fontWeight": 700}),
                html.Div(APP_SUBTITLE, style={**MUTED, "marginTop": "4px"}),
            ],
            style={"marginBottom": "18px"},
        ),

        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Filters", style={"fontWeight": 650, "marginBottom": "10px"}),

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

                                html.Div(style={"height": "10px"}),

                                dbc.Label("Year range"),
                                dcc.RangeSlider(
                                    id="year_range",
                                    min=min_year,
                                    max=max_year,
                                    value=[min_year, max_year],
                                    marks={min_year: str(min_year), max_year: str(max_year)},
                                    allowCross=False,
                                ),

                                dbc.Label("Sex", style={"marginTop": "12px"}),
                                dcc.Dropdown(
                                    id="sex",
                                    options=[{"label": s, "value": s} for s in sex_values],
                                    value="All",
                                    clearable=False,
                                ),

                                dbc.Label("Age group", style={"marginTop": "10px"}),
                                dcc.Dropdown(
                                    id="age_group",
                                    options=[{"label": a, "value": a} for a in age_values],
                                    value="All",
                                    clearable=False,
                                ),

                                dbc.Label("Map year", style={"marginTop": "14px"}),
                                dcc.Slider(
                                    id="map_year",
                                    min=min_year,
                                    max=max_year,
                                    value=max_year,
                                    marks={min_year: str(min_year), max_year: str(max_year)},
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, "height": "100%"},
                    ),
                    md=3,
                ),

                # Main content
                dbc.Col(
                    [
                        # KPI row
                        dbc.Row(
                            [
                                dbc.Col(kpi_card("Latest A", "—", " "), md=2, id="kpi_latest_a"),
                                dbc.Col(kpi_card("Latest B", "—", " "), md=2, id="kpi_latest_b"),
                                dbc.Col(kpi_card("Difference (A - B)", "—", "Compared at latest year"), md=2, id="kpi_diff"),
                                dbc.Col(kpi_card("Change A", "—", f"{min_year} to {max_year}"), md=2, id="kpi_change_a"),
                                dbc.Col(kpi_card("Change B", "—", f"{min_year} to {max_year}"), md=2, id="kpi_change_b"),
                                dbc.Col(kpi_card("National rank (latest)", "—", f"Year {max_year}"), md=2, id="kpi_rank"),
                            ],
                            className="g-3",
                            style={"marginBottom": "16px"},
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Smoking prevalence trend (Region A vs Region B)", style=SECTION_TITLE),
                                        dcc.Graph(id="trend_graph", config=GRAPH_CONFIG, style={"height": "420px"}),
                                    ],
                                    md=7,
                                ),
                                dbc.Col(
                                    [
                                        html.Div(id="map_title", style=SECTION_TITLE),
                                        dcc.Graph(id="map_graph", config=GRAPH_CONFIG, style={"height": "420px"}),
                                    ],
                                    md=5,
                                ),
                            ],
                            className="g-4",
                        ),

                        html.Hr(style={"margin": "18px 0"}),

                        # Evidence Panel
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Evidence panel", style={"fontSize": "24px", "fontWeight": 700, "marginBottom": "6px"}),
                                    html.Div(
                                        "Evidence is computed from your current filters (Region A, Sex, Age group). "
                                        "Use it to justify comparisons with numbers, not vibes.",
                                        style={**MUTED, "marginBottom": "14px"},
                                    ),

                                    dbc.Accordion(
                                        [
                                            dbc.AccordionItem(
                                                [
                                                    html.Div(id="corr_title", style={"fontWeight": 650, "marginBottom": "10px"}),
                                                    dcc.Graph(id="corr_bar", config=GRAPH_CONFIG, style={"height": "300px"}),
                                                ],
                                                title="Correlations",
                                            ),
                                            dbc.AccordionItem(
                                                [
                                                    html.Div(id="reg_title", style={"fontWeight": 650, "marginBottom": "10px"}),
                                                    dcc.Graph(id="reg_coef", config=GRAPH_CONFIG, style={"height": "300px"}),
                                                    html.Div(id="reg_summary", style={"marginTop": "10px"}),
                                                ],
                                                title="Regression evidence",
                                            ),
                                        ],
                                        start_collapsed=True,
                                    ),
                                ]
                            ),
                            style=CARD_STYLE,
                        ),

                        html.Hr(style={"margin": "18px 0"}),

                        # AI Assistant
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("AI Assistant", style={"fontSize": "26px", "fontWeight": 750, "marginBottom": "6px"}),
                                    html.Div(
                                        "Ask a question and get an explanation grounded in your current filters. "
                                        'Example: "Why is Region A higher than Region B?" or "What should I check next?"',
                                        style={**MUTED, "marginBottom": "10px"},
                                    ),

                                    dbc.Alert(id="ai_status", color="secondary", children="", is_open=False),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Input(
                                                    id="ai_question",
                                                    type="text",
                                                    placeholder="Type your question...",
                                                    style={"width": "100%", "padding": "12px", "borderRadius": "10px"},
                                                ),
                                                md=9,
                                            ),
                                            dbc.Col(
                                                dbc.Button("Send", id="ai_send", color="primary", style={"width": "100%", "padding": "12px"}),
                                                md=3,
                                            ),
                                        ],
                                        className="g-2",
                                    ),

                                    html.Div(id="ai_answer", style={"whiteSpace": "pre-wrap", "marginTop": "12px"}),
                                    html.Div(
                                        "Note: The assistant activates when OPENAI_API_KEY is set in Railway Variables. "
                                        f"Model: {OPENAI_MODEL}",
                                        style={**MUTED, "marginTop": "8px", "fontSize": "13px"},
                                    ),
                                ]
                            ),
                            style=CARD_STYLE,
                        ),

                        html.Div(style={"height": "20px"}),
                        html.Div("Built with Dash. Deployed on Railway.", style={**MUTED, "fontSize": "13px"}),
                    ],
                    md=9,
                ),
            ],
            className="g-4",
        ),
    ],
)


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
@app.callback(
    Output("kpi_latest_a", "children"),
    Output("kpi_latest_b", "children"),
    Output("kpi_diff", "children"),
    Output("kpi_change_a", "children"),
    Output("kpi_change_b", "children"),
    Output("kpi_rank", "children"),
    Output("trend_graph", "figure"),
    Output("map_graph", "figure"),
    Output("map_title", "children"),
    Output("corr_title", "children"),
    Output("corr_bar", "figure"),
    Output("reg_title", "children"),
    Output("reg_coef", "figure"),
    Output("reg_summary", "children"),
    Input("region_a", "value"),
    Input("region_b", "value"),
    Input("year_range", "value"),
    Input("sex", "value"),
    Input("age_group", "value"),
    Input("map_year", "value"),
)
def update_dashboard(region_a, region_b, year_range, sex, age_group, map_year):
    y0, y1 = int(year_range[0]), int(year_range[1])

    # KPI numbers
    a_latest, a_y = region_latest_prevalence(df, region_a, sex=sex, age_group=age_group)
    b_latest, b_y = region_latest_prevalence(df, region_b, sex=sex, age_group=age_group)

    latest_year_used = int(max(a_y, b_y)) if not pd.isna(a_y) and not pd.isna(b_y) else max_year
    diff = (a_latest - b_latest) if (not pd.isna(a_latest) and not pd.isna(b_latest)) else np.nan

    a_change = change_over_period(df, region_a, y0, y1, sex=sex, age_group=age_group)
    b_change = change_over_period(df, region_b, y0, y1, sex=sex, age_group=age_group)

    a_peak, a_peak_year = peak_value(df, region_a, sex=sex, age_group=age_group)
    b_peak, b_peak_year = peak_value(df, region_b, sex=sex, age_group=age_group)

    a_rank, rank_year = national_rank_latest(df, region_a, sex=sex, age_group=age_group)
    b_rank, _ = national_rank_latest(df, region_b, sex=sex, age_group=age_group)

    # KPI cards
    kpi_a = kpi_card(
        f"Latest A ({region_a})",
        f"{a_latest:.2f}%" if not pd.isna(a_latest) else "—",
        f"Year {a_y} • Peak {a_peak:.2f}% ({a_peak_year})" if not pd.isna(a_y) else " ",
    )
    kpi_b = kpi_card(
        f"Latest B ({region_b})",
        f"{b_latest:.2f}%" if not pd.isna(b_latest) else "—",
        f"Year {b_y} • Peak {b_peak:.2f}% ({b_peak_year})" if not pd.isna(b_y) else " ",
    )
    kpi_d = kpi_card(
        "Difference (A - B)",
        f"{diff:+.2f} pp" if not pd.isna(diff) else "—",
        "Compared at latest available year",
    )
    kpi_ca = kpi_card(
        f"Change A ({region_a})",
        f"{a_change:+.2f} pp" if not pd.isna(a_change) else "—",
        f"{y0} to {y1}",
    )
    kpi_cb = kpi_card(
        f"Change B ({region_b})",
        f"{b_change:+.2f} pp" if not pd.isna(b_change) else "—",
        f"{y0} to {y1}",
    )
    kpi_r = kpi_card(
        "National rank (latest)",
        f"A #{a_rank} | B #{b_rank}" if (not pd.isna(a_rank) and not pd.isna(b_rank)) else "—",
        f"Year {rank_year}",
    )

    # Trend figure
    d_trend = apply_filters(df, year_min=y0, year_max=y1, sex=sex, age_group=age_group)
    d_a = d_trend[d_trend["region"] == region_a].groupby("year", as_index=False)["prevalence"].mean()
    d_b = d_trend[d_trend["region"] == region_b].groupby("year", as_index=False)["prevalence"].mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=d_a["year"], y=d_a["prevalence"], mode="lines+markers", name=region_a))
    fig_trend.add_trace(go.Scatter(x=d_b["year"], y=d_b["prevalence"], mode="lines+markers", name=region_b))

    fig_trend.update_layout(
        margin=dict(l=55, r=20, t=20, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="year",
        yaxis_title="Prevalence",
        hovermode="x unified",
    )

    # Map figure
    d_map = apply_filters(df, year_min=map_year, year_max=map_year, sex=sex, age_group=age_group)
    d_map = d_map.groupby("region", as_index=False)["prevalence"].mean()

    fig_map = px.choropleth(
        d_map,
        geojson=italy_geo,
        locations="region",
        featureidkey="properties.reg_name",
        color="prevalence",
        hover_name="region",
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    map_title = f"Smoking prevalence map ({map_year})"

    # Evidence panel
    corr = corr_table(df, region_a, sex=sex, age_group=age_group)
    corr_title = f"Correlation signals for {region_a} (with current filters)"

    if corr.empty:
        fig_corr = go.Figure()
        fig_corr.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    else:
        fig_corr = px.bar(corr, x="correlation", y="feature", orientation="h")
        fig_corr.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="correlation")

    reg = linear_regression_numpy(df, region_a, sex=sex, age_group=age_group)
    reg_title = f"Regression evidence for {region_a} (OLS via numpy)"
    coef_df = reg["coefficients"]
    r2 = reg["r2"]

    if coef_df.empty:
        fig_coef = go.Figure()
        fig_coef.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        reg_summary = "No regression model could be estimated for the current filters (missing columns or too few rows)."
    else:
        fig_coef = px.bar(coef_df, x="coefficient", y="feature", orientation="h")
        fig_coef.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="coefficient")
        reg_summary = evidence_summary_text(region_a, r2, coef_df)

    return (
        kpi_a, kpi_b, kpi_d, kpi_ca, kpi_cb, kpi_r,
        fig_trend, fig_map, map_title,
        corr_title, fig_corr,
        reg_title, fig_coef, dcc.Markdown(reg_summary),
    )


@app.callback(
    Output("ai_status", "children"),
    Output("ai_status", "is_open"),
    Output("ai_status", "color"),
    Output("ai_answer", "children"),
    Input("ai_send", "n_clicks"),
    State("ai_question", "value"),
    State("region_a", "value"),
    State("region_b", "value"),
    State("year_range", "value"),
    State("sex", "value"),
    State("age_group", "value"),
    prevent_initial_call=True,
)
def handle_ai(n, question, region_a, region_b, year_range, sex, age_group):
    if not question or not question.strip():
        return "Type a question first.", True, "warning", ""

    y0, y1 = int(year_range[0]), int(year_range[1])

    a_latest, a_y = region_latest_prevalence(df, region_a, sex=sex, age_group=age_group)
    b_latest, b_y = region_latest_prevalence(df, region_b, sex=sex, age_group=age_group)
    diff = (a_latest - b_latest) if (not pd.isna(a_latest) and not pd.isna(b_latest)) else np.nan

    reg = linear_regression_numpy(df, region_a, sex=sex, age_group=age_group)
    coef_df = reg["coefficients"].head(4)
    r2 = reg["r2"]

    context = (
        f"Filters: Region A={region_a}, Region B={region_b}, Years={y0}-{y1}, Sex={sex}, Age={age_group}\n"
        f"Latest A: {a_latest:.2f}% (year {a_y})\n"
        f"Latest B: {b_latest:.2f}% (year {b_y})\n"
        f"Diff (A-B): {diff:+.2f} pp\n"
        f"Regression (Region A) R2: {r2:.2f}\n"
        f"Top coefficients:\n{coef_df.to_string(index=False)}\n"
    )

    ok, text = ai_answer(question.strip(), context)

    if ok:
        return "AI Assistant is active.", True, "success", text
    else:
        return text, True, "secondary", ""


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
