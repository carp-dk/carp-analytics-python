#!/usr/bin/env python3
"""
Example script demonstrating basic usage of the carp-analytics-python library.

Run from the project root after installing the package:
    python examples/main.py data/study/data-streams.json
"""

from carp import CarpDataStream
import sys

def main():
    file_path = "data/study/data-streams.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    print(f"Loading {file_path}...")
    data = CarpDataStream(file_path)
    
    # Scan and print schema
    print("Scanning schema...")
    data.print_schema()
    
    # Example: Grouping data by data type
    # output_dir = "output_groups"
    # print(f"Grouping data into {output_dir}...")
    # data.group_by_field("dataStream.dataType.name", output_dir)

    # Convert to Parquet
    parquet_dir = "output_parquet"
    data.convert_to_parquet(parquet_dir)
    
    # Load back as DataFrame
    df = data.get_dataframe("dk.cachet.carp.stepcount", parquet_dir)
    if df is not None:
        print(f"Loaded {len(df)} stepcount records.")
        print(df.head())


if __name__ == '__main__':
    main()