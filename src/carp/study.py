"""Composition root for the modular CARP Analytics API."""

from __future__ import annotations

from pathlib import Path

from carp.constants import PARTICIPANT_FILE
from carp.core.files import resolve_paths
from carp.export import ExportService
from carp.frames import FrameService
from carp.participants import ParticipantDirectory, ParticipantService
from carp.plotting import PlotService
from carp.records import RecordService
from carp.schema import SchemaService
from carp.types import TypeDefinitionService


def _discover_participant_folders(file_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return phase folders that contain participant metadata."""

    folders = {path.parent for path in file_paths if (path.parent / PARTICIPANT_FILE).exists()}
    return tuple(sorted(folders))


class CarpStudy:
    """Primary public entrypoint for working with CARP study data."""

    def __init__(
        self,
        file_paths: str | Path | tuple[str | Path, ...] | list[str | Path],
        load_participants: bool = True,
    ):
        self.file_paths = resolve_paths(file_paths)
        participant_folders = _discover_participant_folders(self.file_paths) if load_participants else ()
        self._directory = ParticipantDirectory.from_folders(participant_folders)
        self.records = RecordService(self.file_paths, self._directory)
        self.participants = ParticipantService(self, self._directory)
        self.schema = SchemaService(self.records)
        self.export = ExportService(self.records)
        self.frames = FrameService(self.records, self._directory)
        self.types = TypeDefinitionService(self.records)
        self.plots = PlotService(self.frames, self.participants)

    def participant(self, email: str) -> object:
        """Return a participant-scoped view by email."""

        return self.participants.view(email)
