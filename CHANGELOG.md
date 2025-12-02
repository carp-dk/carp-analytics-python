# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-12-02

### Added

- Initial release of CARP Analytics Python library
- `CarpDataStream` class for loading and processing CARP data streams
- Streaming JSON parsing with `ijson` for memory-efficient processing
- Schema discovery and inference from data
- Parquet export for faster subsequent analysis
- `ParticipantManager` for tracking participants across study phases
- `ParticipantAccessor` fluent API for accessing individual participant data
- `ParticipantInfo` dataclass for participant metadata
- Data grouping by field, participant, email, SSN, or name
- DataFrame integration with pandas
- Location visualization with Folium heatmaps
- Rich terminal output with progress bars and formatted tables
- CLI entry point (`carp` command) for command-line usage
- Support for Python 3.10, 3.11, 3.12, and 3.13
- Optional dependencies for pandas, visualization, and scientific computing
- Type hints with PEP 561 py.typed marker

### Dependencies

- Core: ijson, rich, tqdm
- Optional pandas: pandas, pyarrow
- Optional viz: matplotlib, folium
- Optional science: numpy, scipy, scikit-learn

[Unreleased]: https://github.com/carp-dk/carp-analytics-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/carp-dk/carp-analytics-python/releases/tag/v0.1.0
