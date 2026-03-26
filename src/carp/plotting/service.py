"""High-level plotting service for study and participant data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carp.constants import DEFAULT_LOCATION_TYPE, DEFAULT_STEP_TYPE
from carp.core.naming import sanitize_filename

from .prepare import candidate_series, frames_from_items, prepare_location_frame, prepare_step_frame
from .render import render_heatmap


class PlotService:
    """Render HTML maps from study data or typed objects."""

    def __init__(self, frames: Any, participants: Any) -> None:
        self._frames = frames
        self._participants = participants
        self.candidate_series = candidate_series

    def participant(
        self,
        email: str,
        output_file: str | None = None,
        location_type: str = DEFAULT_LOCATION_TYPE,
        step_type: str = DEFAULT_STEP_TYPE,
        parquet_dir: str | None = None,
        include_steps: bool = True,
    ) -> str | None:
        """Render a participant heatmap from an email address."""

        view = self._participants.view(email)
        if not view.exists:
            return None
        default_name = sanitize_filename(email.replace("@", "_at_"), allowed="-_.")
        return self._plot_for_deployments(
            view.deployment_ids,
            output_file or f"{default_name}_location.html",
            location_type,
            step_type,
            parquet_dir,
            include_steps,
        )

    def deployment(
        self,
        deployment_id: str,
        output_file: str = "deployment_heatmap.html",
        location_type: str = DEFAULT_LOCATION_TYPE,
        step_type: str = DEFAULT_STEP_TYPE,
        parquet_dir: str | None = None,
        include_steps: bool = True,
    ) -> str | None:
        """Render a heatmap for a single deployment."""

        return self._plot_for_deployments(
            (deployment_id,),
            output_file,
            location_type,
            step_type,
            parquet_dir,
            include_steps,
        )

    def unified(
        self,
        unified_id: str,
        output_file: str = "participant_heatmap.html",
        location_type: str = DEFAULT_LOCATION_TYPE,
        step_type: str = DEFAULT_STEP_TYPE,
        parquet_dir: str | None = None,
        include_steps: bool = True,
    ) -> str | None:
        """Render a heatmap for a unified participant."""

        deployment_ids = tuple(participant.study_deployment_id for participant in self._participants.unified(unified_id))
        if not deployment_ids:
            return None
        return self._plot_for_deployments(
            deployment_ids,
            output_file,
            location_type,
            step_type,
            parquet_dir,
            include_steps,
        )

    def from_items(
        self,
        location_items: list[Any],
        step_items: list[Any] | None = None,
        output_file: str = "user_heatmap.html",
    ) -> str | None:
        """Render a heatmap from type-safe Python objects."""

        location_frame, step_frame = frames_from_items(location_items, step_items)
        return render_heatmap(location_frame, step_frame, output_file)

    def _plot_for_deployments(
        self,
        deployment_ids: tuple[str, ...],
        output_file: str,
        location_type: str,
        step_type: str,
        parquet_dir: str | None,
        include_steps: bool,
    ) -> str | None:
        """Render a heatmap for a set of deployments."""

        location_frame = self._frames.get_dataframe(location_type, parquet_dir)
        if location_frame.empty:
            return None
        location_ids = candidate_series(location_frame, ["studyDeploymentId", "dataStream.studyDeploymentId"])
        if location_ids is None:
            return None
        filtered_location = prepare_location_frame(location_frame[location_ids.isin(deployment_ids)])
        if filtered_location.empty:
            return None
        if not include_steps:
            return render_heatmap(filtered_location, filtered_location.iloc[0:0], output_file)
        step_frame = self._frames.get_dataframe(step_type, parquet_dir)
        if step_frame.empty:
            return render_heatmap(filtered_location, step_frame, output_file)
        step_ids = candidate_series(step_frame, ["studyDeploymentId", "dataStream.studyDeploymentId"])
        if step_ids is None:
            return render_heatmap(filtered_location, step_frame.iloc[0:0], Path(output_file))
        filtered_steps = prepare_step_frame(step_frame[step_ids.isin(deployment_ids)])
        return render_heatmap(filtered_location, filtered_steps, Path(output_file))
