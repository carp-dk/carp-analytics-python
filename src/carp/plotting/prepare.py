"""Plot-data preparation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from carp.core.dependencies import import_or_raise


def _extract_part(value: Any, part: str) -> Any:
    """Extract one nested key from a dictionary value."""

    return value.get(part) if isinstance(value, dict) else None


def candidate_series(frame: Any, candidates: Iterable[str]) -> Any:
    """Return the first matching dataframe series for the given candidates."""

    for path in candidates:
        if path in frame.columns:
            return frame[path]
        parts = path.split(".")
        if parts[0] not in frame.columns:
            continue
        series = frame[parts[0]]
        for part in parts[1:]:
            series = series.apply(_extract_part, args=(part,))
        return series
    return None


def prepare_location_frame(frame: Any) -> Any:
    """Add normalized plotting columns to a location dataframe."""

    location = frame.copy()
    location["_lat"] = candidate_series(location, ["measurement.data.latitude", "latitude"])
    location["_lon"] = candidate_series(location, ["measurement.data.longitude", "longitude"])
    location["_time"] = candidate_series(
        location,
        ["measurement.sensorStartTime", "sensorStartTime"],
    )
    return location


def prepare_step_frame(frame: Any) -> Any:
    """Add normalized plotting columns to a step dataframe."""

    steps = frame.copy()
    steps["_steps"] = candidate_series(steps, ["measurement.data.steps", "steps"])
    steps["_time"] = candidate_series(steps, ["measurement.sensorStartTime", "sensorStartTime"])
    return steps


def frames_from_items(location_items: list[Any], step_items: list[Any] | None = None) -> tuple[Any, Any]:
    """Build plotting dataframes from type-safe objects."""

    pandas = import_or_raise("pandas", "viz")

    def attr_path(value: Any, path: str) -> Any:
        current = value
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    location_rows = []
    for item in location_items:
        latitude = attr_path(item, "measurement.data.latitude")
        longitude = attr_path(item, "measurement.data.longitude")
        timestamp = attr_path(item, "measurement.sensorStartTime")
        if latitude is not None and longitude is not None:
            location_rows.append({"_lat": latitude, "_lon": longitude, "_time": timestamp})
    step_rows = []
    for item in step_items or []:
        steps = attr_path(item, "measurement.data.steps")
        timestamp = attr_path(item, "measurement.sensorStartTime")
        if steps is not None:
            step_rows.append({"_steps": steps, "_time": timestamp})
    return pandas.DataFrame(location_rows), pandas.DataFrame(step_rows)
