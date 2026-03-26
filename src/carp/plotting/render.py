"""HTML map rendering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carp.core.dependencies import import_or_raise


def render_heatmap(location_frame: Any, step_frame: Any, output_path: str | Path) -> str | None:
    """Render a heatmap and optional step markers to HTML."""

    pandas = import_or_raise("pandas", "viz")
    folium = import_or_raise("folium", "viz")
    heatmap = import_or_raise("folium.plugins", "viz").HeatMap
    if {"_lat", "_lon"} - set(location_frame.columns):
        return None
    location = location_frame.dropna(subset=["_lat", "_lon"])
    if location.empty:
        return None
    map_view = folium.Map(location=[location["_lat"].mean(), location["_lon"].mean()], zoom_start=12)
    heatmap(location[["_lat", "_lon"]].values.tolist()).add_to(map_view)
    if not step_frame.empty and {"_steps", "_time"} <= set(step_frame.columns):
        merged = _merge_steps(pandas, location, step_frame)
        for _, row in merged.iterrows():
            if row["_steps"] and pandas.notnull(row["_lat"]) and pandas.notnull(row["_lon"]):
                folium.CircleMarker(
                    location=[row["_lat"], row["_lon"]],
                    radius=min(max(row["_steps"] / 10, 3), 20),
                    popup=f"Steps: {row['_steps']}<br>Time: {row['_time']}",
                    color="blue",
                    fill=True,
                    fill_color="blue",
                ).add_to(map_view)
    path = Path(output_path)
    map_view.save(path)
    return str(path)


def _merge_steps(pandas: Any, location: Any, step_frame: Any) -> Any:
    """Merge step markers onto the nearest location timestamps."""

    if "_time" not in location.columns or "_time" not in step_frame.columns:
        return step_frame.iloc[0:0]
    sorted_location = location.sort_values("_time").copy()
    sorted_steps = step_frame.sort_values("_time").copy()
    sorted_location["_time"] = sorted_location["_time"].astype("int64")
    sorted_steps["_time"] = sorted_steps["_time"].astype("int64")
    return pandas.merge_asof(
        sorted_steps,
        sorted_location[["_time", "_lat", "_lon"]],
        on="_time",
        direction="nearest",
        tolerance=300_000_000,
    )
