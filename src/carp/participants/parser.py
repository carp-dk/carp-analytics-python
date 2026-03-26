"""Parsing helpers for `participant-data.json` files."""

from __future__ import annotations

import json
from pathlib import Path

from carp.core.models import ParticipantInfo


def _coerce_full_name(value: object) -> str | None:
    """Normalize CARP full-name payloads."""

    if isinstance(value, str):
        return value or None
    if not isinstance(value, dict):
        return None
    parts = [value.get(key) for key in ("firstName", "middleName", "lastName")]
    clean = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return " ".join(clean) or None


def _coerce_ssn(value: object) -> str | None:
    """Normalize CARP SSN payloads."""

    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        nested = value.get("socialSecurityNumber")
        return str(nested) if nested else None
    return None


def _apply_consent(participant: ParticipantInfo, value: object) -> None:
    """Populate consent-related participant fields."""

    if not isinstance(value, dict):
        return
    participant.consent_signed = True
    participant.consent_timestamp = value.get("signedTimestamp")
    participant.user_id = value.get("userId")
    participant.email = value.get("name")
    if participant.full_name:
        return
    consent_payload = value.get("consent")
    if not isinstance(consent_payload, str):
        return
    try:
        signature = json.loads(consent_payload).get("signature", {})
    except json.JSONDecodeError:
        return
    first_name = (signature.get("firstName") or "").strip()
    last_name = (signature.get("lastName") or "").strip()
    participant.full_name = f"{first_name} {last_name}".strip() or None


def load_participant_file(file_path: Path) -> dict[str, ParticipantInfo]:
    """Load participant records from a single phase folder."""

    participants: dict[str, ParticipantInfo] = {}
    data = json.loads(file_path.read_text(encoding="utf-8"))
    for entry in data:
        deployment_id = entry.get("studyDeploymentId")
        if not deployment_id:
            continue
        for role in entry.get("roles", []):
            info = ParticipantInfo(
                study_deployment_id=deployment_id,
                role_name=role.get("roleName", "Participant"),
                source_folder=file_path.parent.name,
            )
            role_data = role.get("data", {})
            info.full_name = _coerce_full_name(role_data.get("dk.carp.webservices.input.full_name"))
            info.sex = role_data.get("dk.cachet.carp.input.sex")
            info.ssn = _coerce_ssn(role_data.get("dk.carp.webservices.input.ssn"))
            _apply_consent(info, role_data.get("dk.carp.webservices.input.informed_consent"))
            participants[deployment_id] = info
    return participants
