"""
Chip Stocks Compared: an interactive Plotly Dash dashboard comparing NVIDIA, AMD and Intel
(with TSMC, Broadcom and Qualcomm available in the company filter).

Md Jonayed Hossain Chowdhury. Developed for CPS*5745.

Run locally:   pip install -r requirements.txt && python app.py   ->  http://127.0.0.1:8050
On Vercel, pyproject.toml points the Python runtime at the Flask server `app` in this file.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

import metrics as M

AUTHOR = "Md Jonayed Hossain Chowdhury"
ROOT = Path(__file__).resolve().parent

NAMES = {"NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel", "TSM": "TSMC", "AVGO": "Broadcom", "QCOM": "Qualcomm"}
ALL = list(NAMES)
DEFAULT = ["NVDA", "AMD", "INTC"]
# Colour follows the company, in a fixed order, never by rank (validated palette, slots 1 to 6).
COLOR = {"NVDA": "#2a78d6", "AMD": "#eb6834", "INTC": "#1baf7a", "TSM": "#eda100", "AVGO": "#e87ba4", "QCOM": "#008300"}

NAVY, INK, MUTED, GRID = "#1f3a5f", "#1b1f24", "#5b6470", "#e8e8e3"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

DF = M.load()
LAST = DF["Date"].max()
FIRST = DF["Date"].min()
SYNTHETIC = (ROOT / "data" / "SYNTHETIC").exists()
PRESETS = {"1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3), "5Y": pd.DateOffset(years=5)}


def preset_range(key):
    if key == "YTD":
        return pd.Timestamp(LAST.year, 1, 1), LAST
    if key == "MAX":
        return FIRST, LAST
    return max(FIRST, LAST - PRESETS[key]), LAST


DEFAULT_START, DEFAULT_END = preset_range("5Y")


# ---------------------------------------------------------------- figure helpers
def base_layout(fig, ytitle, height=340, yfmt=None, legend=True):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=60, r=20, t=10, b=40),
        font=dict(family=FONT, size=12, color=INK), hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0, title=None) if legend else None, showlegend=legend,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zerolinecolor="#c9c9c2", tickformat=yfmt)
    fig.update_xaxes(gridcolor=GRID, showgrid=False)
    return fig


def empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTED), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def fig_performance(close, scale):
    r = M.rebased(close)
    fig = go.Figure()
    for t in r.columns:
        fig.add_trace(go.Scatter(x=r.index, y=r[t], name=f"{NAMES[t]} ({t})", mode="lines",
                                 line=dict(color=COLOR[t], width=2),
                                 hovertemplate=f"{t}: %{{y:,.0f}}<extra></extra>"))
    fig.add_hline(y=100, line=dict(color="#9a988f", dash="dot", width=1))
    base_layout(fig, "value of $100 invested at the start", height=380)
    if scale == "log":
        lo, hi = float(np.nanmin(r.values)), float(np.nanmax(r.values))
        ticks = [v for v in [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000] if lo / 1.5 <= v <= hi * 1.5]
        fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${v:,}" for v in ticks], minor=dict(showgrid=False))
    else:
        fig.update_yaxes(type="linear", tickprefix="$", tickformat=",.0f")
    return fig


def fig_drawdown(close):
    d = M.drawdown(close)
    fig = go.Figure()
    for t in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d[t], name=t, mode="lines", line=dict(color=COLOR[t], width=1.8),
                                 hovertemplate=f"{t}: %{{y:.1f}}%<extra></extra>"))
    base_layout(fig, "below the previous peak (%)")
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    return fig


def fig_volatility(close):
    v = M.rolling_volatility(close, 30)
    fig = go.Figure()
    for t in v.columns:
        fig.add_trace(go.Scatter(x=v.index, y=v[t], name=t, mode="lines", line=dict(color=COLOR[t], width=1.8),
                                 hovertemplate=f"{t}: %{{y:.0f}}%<extra></extra>"))
    base_layout(fig, "30-day volatility, annualised (%)")
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    return fig


def fig_calendar(close):
    c = M.calendar_returns(close)
    fig = go.Figure()
    for t in close.columns:
        s = c[c["Ticker"] == t]
        labels = [f"{y}{'*' if p else ''}" for y, p in zip(s["Year"], s["partial"])]
        fig.add_trace(go.Bar(x=labels, y=s["return"] * 100, name=t, marker_color=COLOR[t],
                             hovertemplate=f"{t} %{{x}}: %{{y:+.1f}}%<extra></extra>"))
    base_layout(fig, "return in the year (%)")
    fig.update_layout(barmode="group", bargap=0.18, bargroupgap=0.05, hovermode="closest")
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinewidth=1.5)
    return fig


def fig_risk_return(close):
    s = M.summary(close)
    fig = go.Figure()
    for t, row in s.iterrows():
        fig.add_trace(go.Scatter(x=[row["volatility"] * 100], y=[row["cagr"] * 100], mode="markers+text", name=t,
                                 text=[t], textposition="top center", textfont=dict(color=INK, size=12),
                                 marker=dict(size=16, color=COLOR[t], line=dict(color="white", width=2)),
                                 hovertemplate=f"<b>{NAMES[t]}</b><br>annual return %{{y:.1f}}%<br>"
                                               f"volatility %{{x:.1f}}%<br>worst drop {row['max_drawdown']*100:.1f}%<extra></extra>"))
    base_layout(fig, "annualised return (%)", legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="annualised volatility (%)", ticksuffix="%", rangemode="tozero", showgrid=True)
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinewidth=1.5)
    return fig


def fig_distribution(close):
    r = M.daily_returns(close) * 100
    fig = go.Figure()
    for t in r.columns:
        fig.add_trace(go.Box(y=r[t].dropna(), name=t, marker_color=COLOR[t], boxmean=True, boxpoints="outliers",
                             marker=dict(size=3, opacity=0.5), line=dict(width=1.5),
                             hovertemplate=f"{t}: %{{y:.2f}}%<extra></extra>"))
    base_layout(fig, "daily return (%)", legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(ticksuffix="%", zeroline=True)
    return fig


def fig_correlation(close):
    c = M.correlation(close)
    labels = list(c.columns)
    # Correlation has a sign, so the scale diverges around a neutral grey at 0.
    fig = go.Figure(go.Heatmap(z=c.values, x=labels, y=labels, zmin=-1, zmax=1, zmid=0,
                               colorscale=[[0, "#b3261e"], [0.5, "#eeeeec"], [1, "#1f3a5f"]],
                               colorbar=dict(title="r", thickness=12, tickvals=[-1, -0.5, 0, 0.5, 1]),
                               text=np.round(c.values, 2), texttemplate="%{text:.2f}",
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=60, r=20, t=10, b=40),
                      font=dict(family=FONT, size=12, color=INK))
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_dollar_volume(df, tickers, start, end):
    dv = M.wide_dollar_volume(df, tickers, start, end)
    monthly = dv.resample("MS").mean() / 1e9
    fig = go.Figure()
    for t in monthly.columns:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[t], name=t, marker_color=COLOR[t],
                             hovertemplate=f"{t} %{{x|%b %Y}}: $%{{y:.2f}}B a day<extra></extra>"))
    base_layout(fig, "average traded value per day ($ billions)")
    fig.update_layout(barmode="group", hovermode="closest", bargap=0.15)
    fig.update_yaxes(tickprefix="$", ticksuffix="B")
    return fig


def kpi_cards(close):
    s = M.summary(close)
    cards = []
    for t, row in s.iterrows():
        up = row["total_return"] >= 0
        cards.append(html.Div(className="kpi", style={"borderTop": f"4px solid {COLOR[t]}"}, children=[
            html.Div(f"{NAMES[t]} · {t}", className="kpi-name"),
            html.Div(f"${row['last']:,.2f}", className="kpi-value"),
            html.Div([html.Span(f"{'▲' if up else '▼'} {row['total_return']*100:+,.1f}%",
                                className="up" if up else "down"),
                      html.Span(f" since {row['start']:%b %d, %Y}", className="kpi-sub")]),
        ]))
    return cards


def key_differences(close):
    """Plain-language findings for the current selection, computed from the same numbers the charts use."""
    s = M.summary(close)
    if len(s) < 2:
        return [html.Li("Select at least two companies to compare.")]
    best, worst = s["total_return"].idxmax(), s["total_return"].idxmin()
    calm, wild = s["volatility"].idxmin(), s["volatility"].idxmax()
    deep = s["max_drawdown"].idxmin()
    c = M.correlation(close)
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(c.columns) for b in c.columns[i + 1:]]
    hi = max(pairs, key=lambda p: p[2]); lo = min(pairs, key=lambda p: p[2])
    ratio = (1 + s.loc[best, "total_return"]) / (1 + s.loc[worst, "total_return"])
    items = [
        html.Li([html.B(f"{NAMES[best]} led, {NAMES[worst]} trailed. "),
                 f"$100 in {best} became ${100*(1+s.loc[best,'total_return']):,.0f}; in {worst}, "
                 f"${100*(1+s.loc[worst,'total_return']):,.0f}. That is {ratio:,.1f} times as much money. "
                 "Use the log scale on the first chart to compare growth rates fairly."]),
        html.Li([html.B("Higher return came with more risk. " if s.loc[wild, "cagr"] >= s.loc[calm, "cagr"]
                        else "More risk did not mean more return. "),
                 f"{wild} was the most volatile ({s.loc[wild,'volatility']*100:.0f}% a year) and {calm} the calmest "
                 f"({s.loc[calm,'volatility']*100:.0f}%), as the risk versus return chart shows."]),
        html.Li([html.B(f"Deepest fall: {NAMES[deep]}. "),
                 f"At its worst, {deep} sat {abs(s.loc[deep,'max_drawdown'])*100:.0f}% below its earlier peak "
                 "(drawdown chart)."]),
        html.Li([html.B("They tend to move together. " if lo[2] >= 0.5 else
                        "Some move together, some do not. " if hi[2] >= 0.5 else
                        "Their daily moves are only loosely related. "),
                 f"{hi[0]} and {hi[1]} have the most similar daily moves (correlation {hi[2]:.2f}); "
                 f"{lo[0]} and {lo[1]} the least ({lo[2]:.2f})."]),
    ]
    return items


# ---------------------------------------------------------------- layout
dash_app = Dash(__name__, title="Chip Stocks Compared · CPS*5745", assets_folder=str(ROOT / "assets"))
# Vercel loads the WSGI object named `app` from app.py, so that name is the Flask server.
app = server = dash_app.server


def card(title, sub, graph_id, extra=None):
    return html.Div(className="card", children=[
        html.Div(className="card-head", children=[html.H3(title), extra] if extra is not None else [html.H3(title)]),
        html.P(sub, className="card-sub"),
        dcc.Graph(id=graph_id, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}),
    ])


dash_app.layout = html.Div(className="page", children=[
    html.Header(className="banner", children=[
        html.Div([html.H1("Chip Stocks Compared"),
                  html.Div("NVIDIA, AMD and Intel: price, risk and how they move together", className="tagline")]),
        html.Div(className="byline", children=[html.Div(AUTHOR, className="author"),
                                               html.Div("Developed for CPS*5745", className="course")]),
    ]),
    html.Div("Development build: synthetic prices, not market data." if SYNTHETIC else "", className="warn",
             style={} if SYNTHETIC else {"display": "none"}),

    html.Section(className="filters", children=[
        html.Div(className="filter", children=[html.Label("Companies", htmlFor="tickers"),
            dcc.Dropdown(id="tickers", options=[{"label": f"{NAMES[t]} ({t})", "value": t} for t in ALL],
                         value=DEFAULT, multi=True, clearable=False)]),
        html.Div(className="filter", children=[html.Label("Date range", htmlFor="dates"),
            dcc.DatePickerRange(id="dates", min_date_allowed=FIRST.date(), max_date_allowed=LAST.date(),
                                start_date=DEFAULT_START.date(), end_date=DEFAULT_END.date(),
                                display_format="MMM D, YYYY")]),
        html.Div(className="filter", children=[html.Label("Quick range"),
            html.Div(className="presets", children=[html.Button(k, id=f"preset-{k}", n_clicks=0, className="preset")
                                                     for k in ["1Y", "3Y", "5Y", "YTD", "MAX"]])]),
        html.Div(className="filter end", children=[html.Button("↺ Reset", id="reset", n_clicks=0, className="reset",
                                                               title="Back to NVIDIA, AMD and Intel, last 5 years")]),
    ]),
    html.Div(id="asof", className="asof"),

    html.Section(id="kpis", className="kpis"),
    html.Section(className="findings", children=[html.H2("Key differences in this selection"), html.Ul(id="findings")]),

    html.Section(className="grid", children=[
        html.Div(className="wide", children=[card(
            "1 · Growth of $100", "Each stock rebased to $100 on the first day of the range, so different share prices compare fairly.",
            "g-perf", dcc.RadioItems(id="scale", options=[{"label": " linear", "value": "linear"},
                                                          {"label": " log scale", "value": "log"}],
                                     value="log", inline=True, className="scale"))]),
        card("2 · Drawdown", "How far each stock was below its highest close so far: the depth and length of its slumps.", "g-dd"),
        card("3 · Volatility", "Swings in daily returns over the previous 30 trading days, annualised. Higher means a bumpier ride.", "g-vol"),
        card("4 · Return by calendar year", "Year-by-year return. * marks a partial year inside the selected range.", "g-cal"),
        card("5 · Risk versus return", "Annualised return against annualised volatility for the whole range. Up and left is better.", "g-rr"),
        card("6 · Daily return distribution", "Box = middle half of trading days; line = median; dashed = mean; dots = unusual days.", "g-dist"),
        card("7 · Correlation of daily returns", "1.00 means two stocks move in lockstep; near 0 means unrelated.", "g-corr"),
        html.Div(className="wide", children=[card(
            "8 · Trading activity", "Average dollar value traded per day, by month (price × shares).", "g-dv")]),
    ]),

    html.Footer(className="foot", children=[
        html.Div([html.B(AUTHOR), " · Developed for CPS*5745 · Kean University · Fall 2026"]),
        html.Div("Data: Yahoo Finance daily prices, adjusted for splits and dividends, refreshed every weekday after the "
                 "US market close by a GitHub Actions job. Returns include dividends. Not investment advice."),
    ]),
])


# ---------------------------------------------------------------- callbacks
@dash_app.callback(Output("dates", "start_date"), Output("dates", "end_date"), Output("tickers", "value"),
              Output("scale", "value"),
              [Input(f"preset-{k}", "n_clicks") for k in ["1Y", "3Y", "5Y", "YTD", "MAX"]] + [Input("reset", "n_clicks")],
              prevent_initial_call=True)
def presets(*_):
    trig = ctx.triggered_id
    if trig == "reset":
        return DEFAULT_START.date(), DEFAULT_END.date(), DEFAULT, "log"
    s, e = preset_range(trig.replace("preset-", ""))
    return s.date(), e.date(), no_update, no_update


@dash_app.callback(
    Output("g-perf", "figure"), Output("g-dd", "figure"), Output("g-vol", "figure"), Output("g-cal", "figure"),
    Output("g-rr", "figure"), Output("g-dist", "figure"), Output("g-corr", "figure"), Output("g-dv", "figure"),
    Output("kpis", "children"), Output("findings", "children"), Output("asof", "children"),
    Input("tickers", "value"), Input("dates", "start_date"), Input("dates", "end_date"), Input("scale", "value"))
def update(tickers, start, end, scale):
    tickers = [t for t in ALL if t in (tickers or [])]  # fixed order keeps colours and legends stable
    start, end = pd.Timestamp(start or DEFAULT_START), pd.Timestamp(end or DEFAULT_END)
    close = M.wide_close(DF, tickers, start, end)
    asof = (f"Showing {len(close):,} trading days, {close.index.min():%b %d, %Y} to {close.index.max():%b %d, %Y}. "
            f"Latest data: {LAST:%A, %B %d, %Y}." if len(close) else "")
    if len(tickers) == 0 or len(close) < 35:
        msg = "Select at least one company and a range of about two months or more."
        return (*[empty(msg)] * 8, [], [html.Li(msg)], asof)
    return (fig_performance(close, scale), fig_drawdown(close), fig_volatility(close), fig_calendar(close),
            fig_risk_return(close), fig_distribution(close), fig_correlation(close),
            fig_dollar_volume(DF, tickers, start, end), kpi_cards(close), key_differences(close), asof)


if __name__ == "__main__":
    dash_app.run(debug=False, port=8050)
