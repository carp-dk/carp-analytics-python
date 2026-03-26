# CARP Analytics Python

[![PyPI version](https://badge.fury.io/py/carp-analytics-python.svg)](https://badge.fury.io/py/carp-analytics-python)
[![Python versions](https://img.shields.io/pypi/pyversions/carp-analytics-python.svg)](https://pypi.org/project/carp-analytics-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`carp-analytics-python` is a Python library for working with CARP study data. It focuses on streaming JSON records, participant lookup, schema discovery, export, parquet conversion, and optional plotting.

## Quick Start

```python
from carp import CarpStudy

study = CarpStudy("sleep-data/phase-1-1/data-streams.json")
print(study.records.count())
print(study.participants.summary_rows()[0])
```

## Main API

`CarpStudy` is the primary entrypoint.

```python
from carp import CarpStudy

study = CarpStudy([
    "sleep-data/phase-1-1/data-streams.json",
    "sleep-data/phase-2-1/data-streams.json",
])

study.schema.scan()
study.export.export_json("output.json", data_type="dk.cachet.carp.stepcount")
study.frames.convert_to_parquet("output_parquet")
study.participant("alice@example.com").info()
```

## CLI

```bash
carp schema sleep-data/phase-1-1/data-streams.json
carp count sleep-data/phase-1-1/data-streams.json
carp participants sleep-data/phase-1-1/data-streams.json
carp export sleep-data/phase-1-1/data-streams.json -o output.json -t dk.cachet.carp.stepcount
carp group sleep-data/phase-1-1/data-streams.json -o grouped_output
carp convert sleep-data/phase-1-1/data-streams.json -o output_parquet
```

## Documentation

The docs are built with Sphinx, `autodoc`, and `napoleon`.

```bash
python -m pip install sphinx sphinx-rtd-theme
sphinx-build -b html docs docs/_build/html
```

## Release Automation

Pushing a new version tag triggers the release workflow. The tag must match the
package version in `pyproject.toml` as either `0.1.0` or `v0.1.0`.

The release workflow reruns tests, linting, type checks, docs builds, and
package builds before it publishes the distributions to PyPI and attaches the
same artifacts to a GitHub release.

PyPI publishing uses GitHub Actions trusted publishing. Configure a trusted
publisher on PyPI for this repository and the `release` workflow, with the
`pypi` environment enabled in GitHub.

## Examples

```bash
python examples/main.py sleep-data/phase-1-1/data-streams.json
python examples/disc.py sleep-data/phase-1-1/data-streams.json
```

## Optional Dependencies

`pandas` and `pyarrow` enable dataframe and parquet support. `folium` enables plotting.
