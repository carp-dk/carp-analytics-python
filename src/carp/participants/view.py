"""Participant-centric study accessors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class ParticipantView:
    """Provide participant-scoped access to study data."""

    def __init__(self, study: Any, email: str):
        self._study = study
        self._email = email

    @property
    def participants(self) -> list[Any]:
        """Return underlying participant models for the view."""

        return list(self._study.participants.by_email(self._email))

    @property
    def deployment_ids(self) -> tuple[str, ...]:
        """Return deployment identifiers for the participant."""

        return tuple(self._study.participants.deployment_ids("email", self._email))

    @property
    def exists(self) -> bool:
        """Return whether the participant exists in the study."""

        return bool(self.participants)

    def info(self) -> dict[str, Any] | None:
        """Return merged participant metadata."""

        if not self.participants:
            return None
        base = self.participants[0]
        return {
            "email": self._email,
            "unified_id": base.unified_participant_id,
            "full_name": base.full_name,
            "ssn": base.ssn,
            "sex": base.sex,
            "user_id": base.user_id,
            "consent_signed": base.consent_signed,
            "consent_timestamp": base.consent_timestamp,
            "folders": sorted({p.source_folder for p in self.participants if p.source_folder}),
            "deployment_ids": sorted(self.deployment_ids),
            "num_deployments": len(self.deployment_ids),
        }

    def iter_records(self, data_type: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield participant records with an optional data-type filter."""

        yield from self._study.records.iter_records(data_type, self.deployment_ids)

    def available_fields(self, sample_size: int = 100) -> list[str]:
        """Return participant-visible field paths."""

        fields: set[str] = set()
        for index, item in enumerate(self.iter_records()):
            if index >= sample_size:
                break
            fields.update(self._study.records.collect_fields(item))
        return sorted(fields)

    def data_types(self) -> list[str]:
        """Return unique data types for the participant."""

        return sorted({self._study.records.data_type(item) for item in self.iter_records()})

    def count(self, data_type: str | None = None) -> int:
        """Return the number of participant records."""

        return sum(1 for _ in self.iter_records(data_type))

    def dataframe(self, data_type: str, parquet_dir: str | None = None) -> Any:
        """Return a dataframe filtered to the participant."""

        frame = self._study.frames.get_dataframe(data_type, parquet_dir)
        if frame is None or frame.empty:
            return frame
        deployment_ids = self._study.plots.candidate_series(
            frame,
            ["studyDeploymentId", "dataStream.studyDeploymentId"],
        )
        return frame if deployment_ids is None else frame[deployment_ids.isin(self.deployment_ids)]

    def plot_location(
        self,
        output_file: str | None = None,
        parquet_dir: str | None = None,
        include_steps: bool = True,
    ) -> str | None:
        """Render a location plot for the participant."""

        result = self._study.plots.participant(
            self._email,
            output_file=output_file,
            parquet_dir=parquet_dir,
            include_steps=include_steps,
        )
        return None if result is None else str(result)
