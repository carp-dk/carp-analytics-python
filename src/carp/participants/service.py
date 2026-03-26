"""High-level participant service for `CarpStudy`."""

from __future__ import annotations

from typing import Any

from .directory import ParticipantDirectory
from .view import ParticipantView


class ParticipantService:
    """Expose participant-centric queries and views."""

    def __init__(self, study: Any, directory: ParticipantDirectory) -> None:
        self._study = study
        self._directory = directory

    def view(self, email: str) -> ParticipantView:
        """Return a participant-scoped view by email."""

        return ParticipantView(self._study, email)

    def by_email(self, email: str) -> list[Any]:
        """Return participant deployments for an email address."""

        return self._directory.find_by_email(email)

    def by_ssn(self, ssn: str) -> list[Any]:
        """Return participant deployments for an SSN."""

        return self._directory.find_by_ssn(ssn)

    def by_name(self, name: str) -> list[Any]:
        """Return participant deployments for a full name."""

        return self._directory.find_by_name(name)

    def deployment_ids(self, field_name: str, value: str) -> tuple[str, ...]:
        """Return deployment identifiers for a participant lookup."""

        return self._directory.deployment_ids(field_name, value)

    def unified(self, unified_id: str) -> list[Any]:
        """Return deployments for a unified participant identifier."""

        return self._directory.get_unified_participant(unified_id)

    def summary_rows(self) -> list[dict[str, str]]:
        """Return participant summary rows for presentation layers."""

        return self._directory.summary_rows()
