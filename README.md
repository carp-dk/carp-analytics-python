# Sleepiness Data Processing Library

A high-performance Python library for processing large JSON data streams from clinical studies.

## Features

- **Streaming JSON Parsing**: Uses `ijson` to handle very large JSON files with minimal memory footprint.
- **Schema Discovery**: Automatically scans and infers the schema of the data.
- **Data Grouping**: Efficiently groups data by any field (e.g., data type, device ID) into separate files.
- **Export**: Export filtered data to JSON.
- **Rich & Tqdm**: Beautiful terminal output and progress bars.

## Installation

```bash
pip install .
```

## Usage

```python
from sleepiness import SleepinessData

# Initialize with a large JSON file
sd = SleepinessData("data/phase-1-1/data-streams.json")

# Scan and print the schema
sd.print_schema()

# Group data by data type into separate files
sd.group_by_field("dataStream.dataType.name", "output_by_type")

# Export specific data type
sd.export_to_json("heartbeat_data.json", data_type="dk.cachet.carp.heartbeat")
```

## Project Structure

- `src/sleepiness/reader.py`: Core logic for streaming and processing JSON data.

