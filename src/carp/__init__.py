"""
CARP Analytics Python - A high-performance library for processing CARP study data.

This library provides tools for streaming, processing, and analysing large JSON
data streams from CARP (Copenhagen Research Platform) clinical and research studies.
"""

from .reader import CarpDataStream, ParticipantManager, ParticipantInfo, ParticipantAccessor

__version__ = "0.1.0"
__author__ = "Copenhagen Research Platform"
__email__ = "support@carp.dk"

__all__ = [
    "CarpDataStream",
    "ParticipantManager",
    "ParticipantInfo",
    "ParticipantAccessor",
    "__version__",
]
