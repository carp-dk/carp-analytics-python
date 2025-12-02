from src.sleepiness import SleepinessData
import sys

def main():
    file_path = "data/phase-2-1/data-streams.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    print(f"Loading {file_path}...")
    sd = SleepinessData(file_path)
    
    # Scan and print schema
    print("Scanning schema...")
    sd.print_schema()
    
    # Example: Grouping data by data type
    # output_dir = "output_groups"
    # print(f"Grouping data into {output_dir}...")
    # sd.group_by_field("dataStream.dataType.name", output_dir)

    # Convert to Parquet
    parquet_dir = "output_parquet"
    sd.convert_to_parquet(parquet_dir)
    
    # Load back as DataFrame
    df = sd.get_dataframe("dk.cachet.carp.stepcount", parquet_dir)
    if df is not None:
        print(f"Loaded {len(df)} stepcount records.")
        print(df.head())


if __name__ == '__main__':
    main()