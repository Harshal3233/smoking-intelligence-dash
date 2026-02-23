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
# Helpers
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


def apply_filters(df: pd.DataFrame, region=None, year_min=None, year_max=None, gender=None, age_group=None) -> pd.DataFrame:
    out = df.copy()

    if region and region != "All":
        out = out[out["region"] == region]

    if year_min is not None:
        out = out[out["year"] >= year_min]
    if year_max is not None:
        out = out[out["year"] <= year_max]

    # Dataset column remains "sex"
    if gender and gender != "All" and "sex" in out.columns:
        out = out[out["sex"] == gender]

    if age_group and age_group != "All" and "age_group" in out.columns:
        out = out[out["age_group"] == age_group]

    return out


def latest_year(df: pd.DataFrame) -> int:
    return int(df["year"].max())


def region_latest_prevalence(df: pd.DataFrame, region: str, gender="All", age_group="All") -> tuple[float, int]:
    d = apply_filters(df, region=region, gender=gender, age_group=age_group)
    if d.empty:
        return (np.nan, np.nan)
    y = latest_year(d)
    v = float(d[d["year"] == y]["prevalence"].mean())
    return v, y


def change_over_period(df: pd.DataFrame, region: str, y0: int, y1: int, gender="All", age_group="All") -> float:
    d = apply_filters(df, region=region, year_min=y0, year_max=y1, gender=gender, age_group=age_group)
    if d.empty:
        return np.nan
    v0 = d[d["year"] == y0]["prevalence"].mean()
    v1 = d[d["year"] == y1]["prevalence"].mean()
    if pd.isna(v0) or pd.isna(v1):
        return np.nan
    return float(v1 - v0)


def peak_value(df: pd.DataFrame, region: str, gender="All", age_group="All") -> tuple[float, int]:
    d = apply_filters(df, region=region, gender=gender, age_group=age_group)
    if d.empty:
        return (np.nan, np.nan)
    idx = d["prevalence"].idxmax()
    return float(d.loc[idx, "prevalence"]), int(d.loc[idx, "year"])


def national_rank_latest(df: pd.DataFrame, region: str, gender="All", age_group="All") -> tuple[int, int]:
    d = apply_filters(df, gender=gender, age_group=age_group)
    if d.empty:
        return (np.nan, np.nan)

    y = latest_year(d)
    latest = d[d["year"] == y].groupby("region", as_index=False)["prevalence"].mean()
    latest["rank"] = latest["prevalence"].rank(ascending=False, method="min").astype(int)

    row = latest[latest["region"] == region]
    if row.empty:
        return (np.nan, y)
    return int(row["rank"].iloc[0]), y


# ------------------------------------------------------------
# OpenAI Helper
# ------------------------------------------------------------
def ai_answer(question: str, context: str) -> tuple[bool, str]:
    if not OPENAI_API_KEY:
        return False, "AI Assistant is not configured."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Answer using only provided context."},
                {"role": "user", "content": f"{context}\n\nQuestion:\n{question}"}
            ],
            temperature=0.2,
            max_tokens=300,
        )

        text = resp.choices[0].message.content.strip()
        return True, text

    except Exception as e:
        return False, f"AI request failed: {str(e)}"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = safe_read_csv(DATA_PATH)
df["year"] = df["year"].astype(int)
italy_geo = safe_read_geojson(GEOJSON_PATH)

regions = sorted(df["region"].dropna().unique().tolist())
gender_values = ["All"] + (sorted(df["sex"].dropna().unique().tolist()) if "sex" in df.columns else [])
age_values = ["All"] + (sorted(df["age_group"].dropna().unique().tolist()) if "age_group" in df.columns else [])

min_year = int(df["year"].min())
max_year = int(df["year"].max())

default_a = regions[0] if regions else "All"
default_b = regions[1] if len(regions) > 1 else default_a


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H2(APP_TITLE),
        html.Div(APP_SUBTITLE, style={"marginBottom": "20px"}),

        dbc.Row([
            dbc.Col([
                dbc.Label("Region A"),
                dcc.Dropdown(id="region_a", options=[{"label": r, "value": r} for r in regions], value=default_a),

                dbc.Label("Region B"),
                dcc.Dropdown(id="region_b", options=[{"label": r, "value": r} for r in regions], value=default_b),

                dbc.Label("Year Range"),
                dcc.RangeSlider(id="year_range", min=min_year, max=max_year,
                                value=[min_year, max_year],
                                marks={min_year: str(min_year), max_year: str(max_year)}),

                dbc.Label("Gender"),
                dcc.Dropdown(id="gender",
                             options=[{"label": g, "value": g} for g in gender_values],
                             value="All"),

                dbc.Label("Age Group"),
                dcc.Dropdown(id="age_group",
                             options=[{"label": a, "value": a} for a in age_values],
                             value="All"),
            ], md=3),

            dbc.Col([
                dcc.Graph(id="trend_graph"),
                html.Hr(),
                html.H4("AI Assistant"),
                dcc.Input(id="ai_question", type="text", placeholder="Ask a question...", style={"width": "80%"}),
                dbc.Button("Send", id="ai_send"),
                html.Div(id="ai_answer", style={"marginTop": "10px"})
            ], md=9)
        ])
    ]
)


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
@app.callback(
    Output("trend_graph", "figure"),
    Input("region_a", "value"),
    Input("region_b", "value"),
    Input("year_range", "value"),
    Input("gender", "value"),
    Input("age_group", "value"),
)
def update_graph(region_a, region_b, year_range, gender, age_group):
    y0, y1 = year_range
    d = apply_filters(df, year_min=y0, year_max=y1, gender=gender, age_group=age_group)

    fig = go.Figure()

    for region in [region_a, region_b]:
        d_r = d[d["region"] == region].groupby("year", as_index=False)["prevalence"].mean()
        fig.add_trace(go.Scatter(x=d_r["year"], y=d_r["prevalence"], mode="lines+markers", name=region))

    fig.update_layout(title="Smoking Prevalence Trend",
                      xaxis_title="Year",
                      yaxis_title="Prevalence")

    return fig


@app.callback(
    Output("ai_answer", "children"),
    Input("ai_send", "n_clicks"),
    State("ai_question", "value"),
    State("region_a", "value"),
    State("region_b", "value"),
    State("year_range", "value"),
    State("gender", "value"),
    State("age_group", "value"),
    prevent_initial_call=True
)
def handle_ai(n, question, region_a, region_b, year_range, gender, age_group):
    if not question:
        return "Please type a question."

    y0, y1 = year_range

    context = (
        f"Filters: Region A={region_a}, Region B={region_b}, "
        f"Years={y0}-{y1}, Gender={gender}, Age={age_group}"
    )

    ok, text = ai_answer(question, context)
    return text


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
