"""Example generated dataclasses for CARP study records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def parse_json_field(value: Any) -> Any:
    """Parse JSON text when a field stores serialized payload data."""

    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@dataclass(slots=True)
class DataType:
    """Data-type metadata for one CARP record."""

    namespace: str | None = None
    name: str | None = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        """Build a data-type object from a dictionary."""

        return obj if not isinstance(obj, dict) else cls(obj.get("namespace"), obj.get("name"))


@dataclass(slots=True)
class DataStream:
    """Stream metadata attached to a CARP record."""

    studyDeploymentId: str | None = None
    deviceRoleName: str | None = None
    dataType: DataType | None = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        """Build stream metadata from a dictionary."""

        if not isinstance(obj, dict):
            return obj
        return cls(
            studyDeploymentId=obj.get("studyDeploymentId"),
            deviceRoleName=obj.get("deviceRoleName"),
            dataType=DataType.from_dict(obj.get("dataType")),
        )


@dataclass(slots=True)
class MeasurementData:
    """Common measurement payload used in the examples."""

    steps: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    response_json: Any = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        """Build a measurement payload from a dictionary."""

        if not isinstance(obj, dict):
            return obj
        return cls(
            steps=obj.get("steps"),
            latitude=obj.get("latitude"),
            longitude=obj.get("longitude"),
            response_json=parse_json_field(obj.get("response_json")),
        )


@dataclass(slots=True)
class Measurement:
    """Measurement wrapper for one CARP record."""

    sensorStartTime: int | None = None
    data: MeasurementData | None = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        """Build a measurement object from a dictionary."""

        if not isinstance(obj, dict):
            return obj
        return cls(
            sensorStartTime=obj.get("sensorStartTime"),
            data=MeasurementData.from_dict(obj.get("data")),
        )


@dataclass(slots=True)
class StudyItem:
    """Example typed CARP record used by the examples notebook."""

    sequenceId: int | None = None
    studyDeploymentId: str | None = None
    deviceRoleName: str | None = None
    triggerIds: list[Any] | None = None
    measurement: Measurement | None = None
    dataStream: DataStream | None = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        """Build a typed study item from a dictionary."""

        if not isinstance(obj, dict):
            return obj
        return cls(
            sequenceId=obj.get("sequenceId"),
            studyDeploymentId=obj.get("studyDeploymentId"),
            deviceRoleName=obj.get("deviceRoleName"),
            triggerIds=obj.get("triggerIds"),
            measurement=Measurement.from_dict(obj.get("measurement")),
            dataStream=DataStream.from_dict(obj.get("dataStream")),
        )
