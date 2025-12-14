"""
GCS Parquet Reader - Efficient row-group based filtering
"""
import gcsfs
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd


def query_parquet_gcs(gcs_path, columns=None, filter_column=None, filter_prefix=None,
                      show_progress=True):
    """
    Read Parquet file from GCS with filtering, processing row groups iteratively.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/file.parquet)
        columns: List of column names to select (None = all)
        filter_column: Column name to filter on
        filter_prefix: String prefix to match (e.g., 'MB' for 'MB%' LIKE query)
        show_progress: Whether to print progress updates

    Returns:
        pandas DataFrame with filtered results
    """
    # Remove gs:// prefix
    if gcs_path.startswith('gs://'):
        gcs_path = gcs_path[5:]

    # Setup GCS filesystem
    fs = gcsfs.GCSFileSystem(token='google_default')

    # Open Parquet file
    with fs.open(gcs_path, 'rb') as f:
        parquet_file = pq.ParquetFile(f)
        total_groups = parquet_file.num_row_groups

        if show_progress:
            print(f"File has {total_groups} row groups, {parquet_file.metadata.num_rows:,} total rows")
            if columns:
                print(f"Selecting columns: {', '.join(columns)}")
            if filter_column and filter_prefix:
                print(f"Filtering: {filter_column} starts with '{filter_prefix}'")
            print()

        result_tables = []

        # Process each row group
        for i in range(total_groups):
            # Read one row group
            table = parquet_file.read_row_group(i, columns=columns)

            # Apply filter if specified
            if filter_column and filter_prefix:
                mask = pc.starts_with(table[filter_column], filter_prefix)
                table = table.filter(mask)

            # Store non-empty results
            if table.num_rows > 0:
                result_tables.append(table)

            # Progress update
            if show_progress and (i + 1) % 100 == 0:
                found = sum(t.num_rows for t in result_tables)
                progress = (i + 1) / total_groups * 100
                print(f"\rProgress: {progress:5.1f}% | {i+1}/{total_groups} groups | Found: {found:,}   ",
                      end='', flush=True)

        if show_progress:
            print()  # New line after progress

        # Combine results
        if not result_tables:
            return pd.DataFrame()

        combined_table = pa.concat_tables(result_tables)
        result_df = combined_table.to_pandas()

        if show_progress:
            print(f"✓ Found {len(result_df):,} rows, {len(result_df.columns)} columns")

        return result_df
