import pandas as pd
import plotly.graph_objects as go

PALETTE = {
    "frontier": "#56B4E9",
    "assets": "#999999",
    "Your portfolio": "#D55E00",
    "Minimum variance": "#009E73",
    "Maximum Sharpe": "#CC79A7",
    "Risk parity": "#E69F00",
}
CHART_HEIGHT = 460
SYMBOLS = {"Your portfolio": "star", "Minimum variance": "diamond", "Maximum Sharpe": "square", "Risk parity": "triangle-up"}


def frontier_figure(
    frontier: pd.DataFrame, assets: pd.DataFrame, portfolios: dict[str, tuple[float, float]]
) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=frontier["volatility"], y=frontier["return"], mode="lines", name="Efficient frontier",
        line=dict(color=PALETTE["frontier"], width=3),
    )
    fig.add_scatter(
        x=assets["volatility"], y=assets["return"], mode="markers", name="Individual assets",
        text=assets.index, hovertemplate="%{text}<br>Volatility %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
        marker=dict(color=PALETTE["assets"], size=6),
    )
    for name, (vol, ret) in portfolios.items():
        add_portfolio_point(fig, name, vol, ret)
    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis=dict(title="Annualised volatility", tickformat=".0%"),
        yaxis=dict(title="Expected annual return", tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=20),
    )
    return fig


def add_portfolio_point(fig: go.Figure, name: str, vol: float, ret: float) -> go.Figure:
    fig.add_scatter(
        x=[vol], y=[ret], mode="markers", name=name,
        marker=dict(color=PALETTE[name], size=14, symbol=SYMBOLS[name], line=dict(color="white", width=1)),
    )
    return fig


def cumulative_figure(oos: pd.DataFrame) -> go.Figure:
    growth = (1 + oos).cumprod()
    fig = go.Figure()
    colours = ["#D55E00", "#999999", "#009E73", "#CC79A7", "#E69F00"]
    for name, colour in zip(growth, colours):
        fig.add_scatter(x=growth.index, y=growth[name] - 1, mode="lines", name=name, line=dict(color=colour, width=2))
    fig.update_layout(
        height=CHART_HEIGHT,
        yaxis=dict(title="Cumulative return (out of sample)", tickformat=".0%"),
        xaxis=dict(title="Week"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=20),
    )
    return fig
