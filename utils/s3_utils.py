import os
import hashlib
from io import BytesIO
from datetime import datetime

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils.config import logger

BOTOCORE_CONFIG = Config(
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

SESSION = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION', 'sa-east-1'),
)
S3_CLIENT = SESSION.client('s3', config=BOTOCORE_CONFIG)

DEFAULT_BUCKET = os.environ.get('S3_BUCKET', 'credit-card-transactions-project')

def _generate_s3_key(
    source: str,
    fmt: str,
    ts: datetime | None = None
) -> str:
    """
    Generate a Hive-style partitioned S3 object key.

    Parameters
    ----------
    source : str
        The source system or dataset name.
    fmt : str
        The file format extension (e.g., 'parquet', 'csv', 'json').
    ts : datetime, optional
        Timestamp to base partitioning on. Defaults to current time.

    Returns
    -------
    str
        The generated S3 key, e.g.
        'raw/source/year=YYYY/month=MM/day=DD/source_YYYYMMDD_HHMMSS.fmt'.
    """
    ts = ts or datetime.now()
    return (
        f"raw/{source}/"
        f"year={ts:%Y}/month={ts:%m}/day={ts:%d}/"
        f"{source}_{ts:%Y%m%d_%H%M%S}.{fmt}"
    )

def _compute_checksum(
    df: pd.DataFrame,
    fmt: str
) -> str:
    """
    Compute an MD5 checksum of a DataFrame serialized in the given format.

    Supports Parquet, CSV, JSON, and text fallback.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to checksum.
    fmt : str
        The serialization format ('parquet', 'csv', 'json', etc.).

    Returns
    -------
    str
        The MD5 checksum of the serialized DataFrame bytes.
    """
    buf = BytesIO()
    fmt = fmt.lower()
    
    if fmt == 'parquet':
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, buf)
    elif fmt == 'csv':
        df.to_csv(buf, index=False)
    elif fmt == 'json':
        df.to_json(buf, orient='records', lines=True)
    else:
        buf.write(df.to_string().encode())
    
    buf.seek(0)
    return hashlib.md5(buf.read()).hexdigest()

def ingest_df_raw_zone_s3(
    df: pd.DataFrame,
    source: str,
    fmt: str,
    bucket: str = DEFAULT_BUCKET,
    extra_metadata: dict[str, str] | None = None
) -> str:
    """
    Ingest a DataFrame into the S3 raw zone preserving 'as-is' format.

    Serializes the DataFrame to the specified format, computes a checksum,
    and uploads it to S3 with Hive-style partitioning and metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        The data to ingest.
    source : str
        The source system or dataset name.
    fmt : str
        The file format ('parquet', 'csv', 'json').
    bucket : str, optional
        The S3 bucket name. Defaults to DEFAULT_BUCKET.
    extra_metadata : dict of str, optional
        Additional metadata to attach to the S3 object.

    Returns
    -------
    str
        The S3 key where the object was uploaded.

    Raises
    ------
    ValueError
        If the specified format is unsupported.
    botocore.exceptions.ClientError
        If the upload to S3 fails.
    """
    if df.empty:
        logger.warning(f"No data for source={source}, skipping ingest.")
        return ''

    key = _generate_s3_key(source, fmt)
    buf = BytesIO()

    fmt = fmt.lower()
    if fmt == 'parquet':
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, buf)
    elif fmt == 'csv':
        df.to_csv(buf, index=False)
    elif fmt == 'json':
        df.to_json(buf, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    buf.seek(0)
    checksum = _compute_checksum(df, fmt)

    metadata = {
        'source': source,
        'rows': str(len(df)),
        'columns': ','.join(df.columns),
        'ingestion_time': datetime.now().isoformat(),
        'checksum': checksum,
        'format': fmt,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    try:
        S3_CLIENT.upload_fileobj(
            Fileobj=buf,
            Bucket=bucket,
            Key=key,
            ExtraArgs={'Metadata': metadata}
        )
        logger.info(f"Ingested raw data to s3://{bucket}/{key}")
        return key
    except ClientError as e:
        logger.error(f"Failed ingestion for {source} to {key}: {e}")
        raise

def list_raw_objects(
    source: str,
    date: datetime | None = None,
    fmt: str = ''
) -> list[str]:
    """
    List objects in the raw zone for a given source and optional date/format filter.

    Parameters
    ----------
    source : str
        The source system or dataset name.
    date : datetime, optional
        The date partition to filter on.
    fmt : str, optional
        File extension filter (e.g., 'parquet', 'csv').

    Returns
    -------
    list of str
        A list of S3 keys matching the criteria.

    Raises
    ------
    botocore.exceptions.ClientError
        If listing objects in S3 fails.
    """
    prefix = f"raw/{source}/"
    if date:
        prefix += f"year={date:%Y}/month={date:%m}/day={date:%d}/"
    try:
        paginator = S3_CLIENT.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=DEFAULT_BUCKET, Prefix=prefix)
        keys = [
            obj['Key']
            for page in pages if 'Contents' in page
            for obj in page['Contents']
            if not fmt or obj['Key'].endswith(f".{fmt}")
        ]
        return keys
    except ClientError as e:
        logger.error(f"Error listing raw objects at {prefix}: {e}")
        raise

def check_raw_object_exists(
    key: str,
    bucket: str = DEFAULT_BUCKET
) -> bool:
    """
    Check if a raw object exists in S3.

    Parameters
    ----------
    key : str
        The S3 object key to check.
    bucket : str, optional
        The S3 bucket name. Defaults to DEFAULT_BUCKET.

    Returns
    -------
    bool
        True if the object exists, False if a 404 error is returned.

    Raises
    ------
    botocore.exceptions.ClientError
        For errors other than 404 (not found).
    """
    try:
        S3_CLIENT.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logger.error(f"Unexpected error checking {key}: {e}")
        raise
