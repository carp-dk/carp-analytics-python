# CARP Analytics Python

[![PyPI version](https://badge.fury.io/py/carp-analytics-python.svg)](https://badge.fury.io/py/carp-analytics-python)
[![Python versions](https://img.shields.io/pypi/pyversions/carp-analytics-python.svg)](https://pypi.org/project/carp-analytics-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Python library for processing and analysing data from [CARP](https://carp.computerome.dk/) (Copenhagen Research Platform) studies.

## Features

- **Schema Discovery**: Automatically scans and infers the schema of the data
- **Data Grouping**: Efficiently groups data by any field (e.g., data type, device ID) into separate files
- **Parquet Export**: Convert JSON data to Parquet for faster subsequent analysis
- **Participant Management**: Link and track participants across multiple study phases
- **Visualization**: Generate location heatmaps and other visualizations
- **Pandas Integration**: Seamlessly work with DataFrames

## Installation

```bash
pip install carp-analytics-python
```

### With Optional Dependencies

```bash
# For pandas/parquet support
pip install carp-analytics-python[pandas]

# For visualization support
pip install carp-analytics-python[viz]

# For scientific computing (numpy, scipy, scikit-learn)
pip install carp-analytics-python[science]

# Install everything
pip install carp-analytics-python[all]
```

### Development Installation

```bash
git clone https://github.com/carp-dk/carp-analytics-python.git
cd carp-analytics-python

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Quick Start

```python
from carp import CarpDataStream

# Initialize with a data file
data = CarpDataStream("data/study-phase-1/data-streams.json")

# Scan and print the schema
data.print_schema()

# Convert to Parquet for faster analysis
data.convert_to_parquet("output_parquet")

# Load data as a DataFrame
df = data.get_dataframe("dk.cachet.carp.stepcount", "output_parquet")
print(df.head())
```

## Working with Participants

```python
from carp import CarpDataStream

# Load data from multiple phases
data = CarpDataStream([
    "data/phase-1/data-streams.json",
    "data/phase-2/data-streams.json",
])

# Print participant summary
data.print_participants()

# Access participant data via email
participant = data.participant("user@example.com")

# Get participant info
print(participant.info())

# Get available data types for this participant
participant.print_data_types()

# Get a DataFrame of step count data
df = participant.dataframe("dk.cachet.carp.stepcount", "output_parquet")
```

## Data Export

```python
# Export specific data type to JSON
data.export_to_json("heartbeat_data.json", data_type="dk.cachet.carp.heartbeat")

# Group data by data type
data.group_by_field("dataStream.dataType.name", "output_by_type")

# Group data by participant
data.group_by_participant("output_by_participant")
```

## Visualization

```python
# Generate location heatmap for a participant
participant = data.participant("user@example.com")
participant.visualize.location(output_file="user_locations.html")
```

## Command Line Interface

The package includes a CLI for common operations:

```bash
# Show schema of data files
carp schema data/study/data-streams.json

# Convert JSON to Parquet
carp convert data/study/data-streams.json -o output_parquet

# Count items in data files
carp count data/study/data-streams.json

# List participants
carp participants data/study/data-streams.json

# Export filtered data
carp export data/study/data-streams.json -o output.json -t dk.cachet.carp.stepcount

# Group data by field
carp group data/study/data-streams.json -f dataStream.dataType.name -o grouped_output
```

## API Reference

### `CarpDataStream`

The main class for working with CARP data streams.

| Method | Description |
|--------|-------------|
| `scan_schema()` | Scan and infer the data schema |
| `print_schema()` | Print the inferred schema as a table |
| `convert_to_parquet(output_dir)` | Convert JSON to Parquet files |
| `get_dataframe(data_type, parquet_dir)` | Load data as a pandas DataFrame |
| `export_to_json(output_path, data_type)` | Export data to JSON file |
| `group_by_field(field_path, output_dir)` | Group data by a specific field |
| `participant(email)` | Access participant data via fluent API |
| `print_participants()` | Print participant summary table |

### `ParticipantAccessor`

Fluent API for accessing individual participant data.

| Method | Description |
|--------|-------------|
| `info()` | Get participant information as a dictionary |
| `print_info()` | Print participant info as a table |
| `all_data(data_type)` | Generator for all participant data |
| `data_types()` | Get all unique data types |
| `dataframe(data_type, parquet_dir)` | Get data as a pandas DataFrame |
| `visualize.location()` | Generate location heatmap |

## Requirements

- Python 3.10+
- ijson (for streaming JSON parsing)
- rich (for terminal output)
- tqdm (for progress bars)

Optional:
- pandas, pyarrow (for DataFrame and Parquet support)
- matplotlib, folium (for visualization)
- numpy, scipy, scikit-learn (for scientific computing)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/featA`)
3. Commit your changes (`git commit -m 'Add some featA'`)
4. Push to the branch (`git push origin feature/featA`)
5. Open a Pull Request

## Licence

This project is licensed under the MIT Licence - see the [Licence](LICENSE) file for details.

## Acknowledgments

- [CARP - Copenhagen Research Platform](https://carp.dk/)
