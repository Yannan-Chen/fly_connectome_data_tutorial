"""
GCS Parquet Lazy Reader - Using Arrow Dataset API for fast filtering
"""
import gcsfs
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd


def query_parquet_lazy(gcs_path, columns=None, filter_column=None, filter_prefix=None):
    """
    Read Parquet from GCS using Arrow Dataset API with predicate pushdown.
    This is much faster than row-group iteration as it uses true lazy evaluation.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/file.parquet)
        columns: List of column names to select (None = all)
        filter_column: Column name to filter on
        filter_prefix: String prefix to match

    Returns:
        pandas DataFrame with filtered results
    """
    # Remove gs:// prefix
    if gcs_path.startswith('gs://'):
        gcs_path = gcs_path[5:]

    # Setup GCS filesystem
    fs = gcsfs.GCSFileSystem(token='google_default')

    print(f"Opening dataset: {gcs_path}")

    # Create Arrow dataset using fsspec-compatible filesystem
    # Use PyArrow's FSSpecHandler to wrap gcsfs
    from pyarrow.fs import FSSpecHandler, PyFileSystem
    pa_fs = PyFileSystem(FSSpecHandler(fs))
    dataset = ds.dataset(gcs_path, filesystem=pa_fs, format='parquet')

    print(f"Dataset has {dataset.count_rows():,} total rows")

    # Build filter expression
    filter_expr = None
    if filter_column and filter_prefix:
        # Use PyArrow compute for filter expression
        filter_expr = pc.starts_with(ds.field(filter_column), filter_prefix)
        print(f"Applying filter: {filter_column} starts with '{filter_prefix}'")

    # Execute query with Arrow's optimized engine
    # This uses predicate pushdown - only matching row groups are read
    print("Executing query...")
    table = dataset.to_table(
        filter=filter_expr,
        columns=columns
    )

    # Convert to pandas
    result_df = table.to_pandas()

    print(f"✓ Loaded {len(result_df):,} rows, {len(result_df.columns)} columns")

    return result_df
