"""Domain models shared by multiple subsystems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ParticipantInfo:
    """Normalized participant metadata for one deployment."""

    study_deployment_id: str
    role_name: str = "Participant"
    full_name: str | None = None
    sex: str | None = None
    ssn: str | None = None
    user_id: str | None = None
    email: str | None = None
    consent_signed: bool = False
    consent_timestamp: str | None = None
    source_folder: str | None = None
    unified_participant_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the participant."""

        return asdict(self)
