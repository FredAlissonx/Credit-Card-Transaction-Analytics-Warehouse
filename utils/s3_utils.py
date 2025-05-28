import os
import hashlib
from io import BytesIO
from datetime import datetime, timedelta
import argparse
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from utils.config import logger

S3_CONFIG = Config(
    retries={
        'max_attempts': int(os.getenv('AWS_MAX_ATTEMPTS', 10)),
        'mode': os.getenv('AWS_RETRY_MODE', 'standard')
    },
    max_pool_connections=int(os.getenv('AWS_MAX_POOL_CONNECTIONS', 100))
)

SESSION = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

s3_client = SESSION.client('s3', config=S3_CONFIG)
DEFAULT_BUCKET = os.getenv('S3_BUCKET', 'credit-card-transactions-project')

@dataclass
class DataLakeLayer:
    RAW: str = "raw"
    BRONZE: str = "bronze"
    SILVER: str = "silver"

class DataQualityError(Exception):
    """Custom exception for data quality issues"""

class DataLakeManager:
    """Base class for data lake operations"""
    def __init__(self, source: str, layer: str, bucket: str = DEFAULT_BUCKET):
        self.source = source
        self.layer = layer
        self.bucket = bucket
        self.s3 = s3_client

    def _generate_key(
        self, fmt: str, processing_date: datetime,
        partition_cols: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate S3 key with Hive-style partitioning"""
        base_path = f"{self.layer}/{self.source}/"
        if partition_cols:
            partition_path = "/".join(f"{k}={v}" for k, v in partition_cols.items())
            return f"{base_path}{partition_path}/{self.source}_{processing_date:%Y%m%d_%H%M%S}.{fmt}"
        return f"{base_path}{self.source}_{processing_date:%Y%m%d_%H%M%S}.{fmt}"

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """No-op validation at base"""
        return df

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data processing to be implemented by subclasses"""
        raise NotImplementedError

    def _compute_checksum(self, df: pd.DataFrame, fmt: str) -> str:
        """Compute MD5 checksum of serialized DataFrame"""
        buf = BytesIO()
        self._serialize_data(df, buf, fmt)
        buf.seek(0)
        return hashlib.md5(buf.read()).hexdigest()

    def _serialize_data(
        self, df: pd.DataFrame, buf: BytesIO, fmt: str
    ):
        """Serialize DataFrame to different formats"""
        fmt = fmt.lower()
        if fmt == 'parquet':
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
        elif fmt == 'csv':
            df.to_csv(buf, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def load_from_previous_layer(
        self, processing_date: datetime, fmt: str = 'parquet'
    ) -> pd.DataFrame:
        """Load data from previous layer with format consideration."""
        prev = {
            DataLakeLayer.BRONZE: DataLakeLayer.RAW,
            DataLakeLayer.SILVER: DataLakeLayer.BRONZE
        }[self.layer]
        prefix = f"{prev}/{self.source}/"

        try:
            resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if 'Contents' not in resp:
                return pd.DataFrame()

            frames = []
            for obj in resp['Contents']:
                if not obj['Key'].endswith(f".{fmt}"):
                    continue
                body = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])['Body'].read()
                if fmt == 'parquet':
                    frames.append(pd.read_parquet(BytesIO(body)))
                elif fmt == 'csv':
                    frames.append(pd.read_csv(BytesIO(body)))
            return pd.concat(frames) if frames else pd.DataFrame()

        except ClientError as e:
            logger.error(f"Error loading from {prev}: {e}")
            raise

    def ingest_data(
        self, df: pd.DataFrame, processing_date: datetime,
        fmt: str = 'parquet',
        partition_cols: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Ingest processed data to target layer"""
        if df.empty:
            logger.warning("No data to ingest")
            return ""
        validated = self._validate_data(df)
        processed = self._process_data(validated)
        key = self._generate_key(fmt, processing_date, partition_cols)
        if self._object_exists(key):
            logger.info(f"Skipping existing object {key}")
            return key
        buf = BytesIO()
        self._serialize_data(processed, buf, fmt)
        buf.seek(0)
        checksum = self._compute_checksum(processed, fmt)
        meta = metadata or {}
        meta.update({
            'source': self.source,
            'layer': self.layer,
            'processing_date': processing_date.isoformat(),
            'checksum': checksum,
            'format': fmt,
            'rows': str(len(processed)),
            'columns': ','.join(processed.columns),
        })
        self.s3.upload_fileobj(Fileobj=buf, Bucket=self.bucket, Key=key, ExtraArgs={'Metadata': meta})
        logger.info(f"Ingested to s3://{self.bucket}/{key}")
        return key

    def _object_exists(self, key: str) -> bool:
        """Check if object exists in S3"""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

class RawLayerManager(DataLakeManager):
    """Raw layer: store data exactly as received."""
    def __init__(self, source: str):
        super().__init__(source, DataLakeLayer.RAW)

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # No-op: raw data is ingested “as-is”
        return df
    
    def ingest_data(
        self, df: pd.DataFrame, processing_date: datetime,
        fmt: str, partition_cols: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        source_key: Optional[str] = None
    ) -> str:
        # For raw, preserve original bytes if source_key provided
        if source_key and fmt.lower() == 'parquet':
            raw_obj = self.s3.get_object(Bucket=self.bucket, Key=source_key)['Body'].read()
            key = self._generate_key(fmt, processing_date, partition_cols)
            meta = metadata or {}
            meta.update({'source': self.source, 'layer': self.layer, 'format': fmt})
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=raw_obj, Metadata=meta)
            logger.info(f"Raw parquet preserved to s3://{self.bucket}/{key}")
            return key
        # fallback to DataFrame serialization
        return super().ingest_data(df, processing_date, fmt, partition_cols, metadata)

class BronzeLayerManager(DataLakeManager):
    def __init__(self, source: str):
        super().__init__(source, DataLakeLayer.BRONZE)

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

class SilverLayerManager(DataLakeManager):
    def __init__(
        self, source: str,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df
    ):
        super().__init__(source, DataLakeLayer.SILVER)
        self.transform_fn = transform_fn

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform_fn(df)


def process_layer(layer: str, source: str, processing_date: datetime, transform_fn: Optional[Callable] = None) -> bool:
    try:
        if layer == DataLakeLayer.RAW:
            # Example usage: ingest pre-loaded df with original format and source_key
            pass  # implement as needed
        elif layer == DataLakeLayer.BRONZE:
            man = BronzeLayerManager(source)
            df = man.load_from_previous_layer(processing_date)
            man.ingest_data(df, processing_date)
        elif layer == DataLakeLayer.SILVER:
            man = SilverLayerManager(source, transform_fn or (lambda df: df))
            df = man.load_from_previous_layer(processing_date)
            man.ingest_data(df, processing_date)
        return True
    except Exception as e:
        logger.error(f"Error processing {layer}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Data Lake Pipeline')
    parser.add_argument('--source', required=True)
    parser.add_argument(
        '--layer', required=True,
        choices=[DataLakeLayer.RAW, DataLakeLayer.BRONZE, DataLakeLayer.SILVER]
    )
    parser.add_argument('--date', help='YYYY-MM-DD')
    args = parser.parse_args()

    date = datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.now() - timedelta(days=1)
    logger.info(f"Starting {args.layer} for {args.source} on {date:%Y-%m-%d}")
    ok = process_layer(args.layer, args.source, date)
    return 0 if ok else 1

if __name__ == '__main__':
    exit(main())
