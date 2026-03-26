# Changelog

## [0.2.0] - 2026-03-26

### Added

- New `CarpStudy` public API as the primary entrypoint for CARP study analysis
- Modular service layout under `carp.core`, `participants`, `records`, `schema`, `export`, `frames`, `types`, `plotting`, and `commandline`
- Self-contained pytest suite with committed multi-phase fixtures and optional `sleep-data` smoke coverage
- 100% line and branch coverage enforcement for `src/carp`
- Sphinx documentation site with autodoc and Napoleon support
- GitHub Actions CI for linting, type-checking, tests, and docs builds
- Tag-driven CD workflow that validates version tags, publishes to PyPI, and creates GitHub releases
- Dedicated `test` and `docs` dependency groups

### Changed

- Replaced the legacy method-heavy design with a thin `CarpStudy` composition root and focused services
- Kept the `carp` CLI command set stable while rewriting the implementation behind modular handlers
- Switched plotting defaults to `dk.cachet.carp.location`
- Made parquet filenames namespace-aware to avoid same-name type collisions
- Added Google-style docstrings and expanded type annotations across the package
- Refreshed the README, example scripts, generated type example, and notebook to use the new API
- Normalized Ruff, MyPy, coverage, and documentation build configuration in `pyproject.toml`

### Removed

- Legacy `carp.reader` monolith
- Legacy `carp.plotting.map_viz` module
- Old `CarpDataStream`-centric example usage and stale plotting/type-generation references

## [0.1.0]

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
