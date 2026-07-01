"""Custom Streamlit component: interactive Plotly chart with draggable alert line.

All line dragging happens in JavaScript (zero rerun). Only the "Register
Alert" button triggers a single rerun to save the alert.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"
_component_func = components.declare_component(
    "price_line_chart",
    path=str(_FRONTEND_DIR),
)


def price_line_chart(
    fig,
    *,
    current_price: float = 0,
    height: int = 700,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Render a Plotly chart with a draggable alert price line.

    The chart is rendered entirely in JavaScript with Plotly.js. A toolbar
    button activates "line mode", which overlays a draggable horizontal line
    on the chart. The user can freely drag it up/down with **zero Streamlit
    reruns**. Only when the user clicks "Register Alert" does the component
    send the price back to Python (one rerun).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to render.
    current_price : float
        Current close price (used for auto direction selection).
    height : int
        Chart height in pixels.
    key : str, optional
        Streamlit widget key.

    Returns
    -------
    dict or None
        ``{"price": float, "direction": str, "_id": str}`` when the user
        clicks "Register Alert". ``None`` at all other times.
    """
    result = _component_func(
        figure_json=fig.to_json(),
        current_price=current_price,
        component_height=height,
        key=key,
        default=None,
    )
    return result
