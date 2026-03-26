"""Participant lookup and unification services."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from carp.constants import PARTICIPANT_FILE
from carp.core.models import ParticipantInfo

from .parser import load_participant_file


def _normalize(value: str | None) -> str | None:
    """Normalize string identifiers for matching."""

    if not value:
        return None
    clean = value.strip().lower()
    return clean or None


class ParticipantDirectory:
    """Store participant metadata across one or more study phases."""

    def __init__(self, participants_by_deployment: dict[str, ParticipantInfo] | None = None):
        self.participants_by_deployment = participants_by_deployment or {}
        self.unified_participants: dict[str, list[ParticipantInfo]] = {}
        self._counter = 0
        if self.participants_by_deployment:
            self._unify()

    @classmethod
    def from_folders(cls, folders: tuple[Path, ...]) -> ParticipantDirectory:
        """Build a participant directory from phase folders."""

        participants: dict[str, ParticipantInfo] = {}
        for folder in folders:
            file_path = folder / PARTICIPANT_FILE
            if file_path.exists():
                participants.update(load_participant_file(file_path))
        return cls(participants)

    def get_participant(self, deployment_id: str) -> ParticipantInfo | None:
        """Return one participant by deployment identifier."""

        return self.participants_by_deployment.get(deployment_id)

    def get_unified_participant(self, unified_id: str) -> list[ParticipantInfo]:
        """Return all deployments for one unified participant."""

        return list(self.unified_participants.get(unified_id, []))

    def find_by_email(self, email: str) -> list[ParticipantInfo]:
        """Find all participant deployments matching an email address."""

        target = _normalize(email)
        return [p for p in self.participants_by_deployment.values() if _normalize(p.email) == target]

    def find_by_ssn(self, ssn: str) -> list[ParticipantInfo]:
        """Find all participant deployments matching an SSN."""

        return [p for p in self.participants_by_deployment.values() if p.ssn == ssn]

    def find_by_name(self, name: str) -> list[ParticipantInfo]:
        """Find all participant deployments matching a full name."""

        target = _normalize(name)
        return [p for p in self.participants_by_deployment.values() if _normalize(p.full_name) == target]

    def deployment_ids(self, field_name: str, value: str) -> tuple[str, ...]:
        """Return deployment identifiers for a participant lookup."""

        matches = getattr(self, f"find_by_{field_name}")(value)
        return tuple(participant.study_deployment_id for participant in matches)

    def summary_rows(self) -> list[dict[str, str]]:
        """Return human-readable participant summary rows."""

        rows: list[dict[str, str]] = []
        for unified_id, participants in self.unified_participants.items():
            folders = sorted({p.source_folder for p in participants if p.source_folder})
            emails = sorted({p.email for p in participants if p.email})
            ssns = sorted({p.ssn for p in participants if p.ssn})
            names = sorted({p.full_name for p in participants if p.full_name})
            rows.append(
                {
                    "unified_id": unified_id,
                    "deployments": str(len(participants)),
                    "folders": ", ".join(folders) or "N/A",
                    "emails": ", ".join(emails) or "N/A",
                    "ssns": ", ".join(ssns) or "N/A",
                    "names": ", ".join(names) or "N/A",
                }
            )
        return rows

    def _register_group(self, participants: list[ParticipantInfo], assigned: set[str]) -> None:
        """Register one unified participant group."""

        unified_id = f"P{self._counter:04d}"
        self._counter += 1
        for participant in participants:
            participant.unified_participant_id = unified_id
            assigned.add(participant.study_deployment_id)
        self.unified_participants[unified_id] = participants

    def _unify(self) -> None:
        """Assign unified participant identifiers across phases."""

        assigned: set[str] = set()
        matchers = ("email", "ssn", "name")
        grouped: dict[str, dict[str, list[ParticipantInfo]]] = {
            "email": defaultdict(list),
            "ssn": defaultdict(list),
            "name": defaultdict(list),
        }
        for participant in self.participants_by_deployment.values():
            if email := _normalize(participant.email):
                grouped["email"][email].append(participant)
            if participant.ssn:
                grouped["ssn"][participant.ssn].append(participant)
            if name := _normalize(participant.full_name):
                grouped["name"][name].append(participant)
        for matcher in matchers:
            for participants in grouped[matcher].values():
                pending = [participant for participant in participants if participant.study_deployment_id not in assigned]
                if pending:
                    self._register_group(pending, assigned)
        for participant in self.participants_by_deployment.values():
            if participant.study_deployment_id not in assigned:
                self._register_group([participant], assigned)
        self._propagate()

    def _propagate(self) -> None:
        """Share the best known metadata across unified deployments."""

        for participants in self.unified_participants.values():
            fields = {
                "full_name": next((p.full_name for p in participants if p.full_name), None),
                "sex": next((p.sex for p in participants if p.sex), None),
                "ssn": next((p.ssn for p in participants if p.ssn), None),
                "email": next((p.email for p in participants if p.email), None),
                "user_id": next((p.user_id for p in participants if p.user_id), None),
                "consent_timestamp": next((p.consent_timestamp for p in participants if p.consent_timestamp), None),
            }
            signed = any(p.consent_signed for p in participants)
            for participant in participants:
                participant.consent_signed = signed
                for field_name, value in fields.items():
                    if value and not getattr(participant, field_name):
                        setattr(participant, field_name, value)
