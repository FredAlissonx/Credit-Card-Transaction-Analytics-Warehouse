"""
S3Utils.py - Utility functions for S3 operations in medallion architecture

This module provides functions for S3 operations used across raw, bronze, silver,
and gold layers in a medallion data architecture.
"""

import boto3
import pandas as pd
from utils.config import logger
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
import os
import json
from datetime import datetime

DEFAULT_BUCKET = os.environ.get('S3_BUCKET', "credit-card-transactions-project")

def get_s3_client():
    try:
        s3_client = boto3.client(
            's3',
            region_name=os.environ.get('AWS_REGION', 'sa-east-1'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        return s3_client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {str(e)}")
        raise

def check_file_exists(
    key: str,
    bucket: str = DEFAULT_BUCKET
) -> bool:
    try:
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            logger.error(f"Error checking if file exists: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error checking if file exists: {str(e)}")
        raise

def list_objects(
    prefix: str = '', 
    suffix: str = '',
    bucket: str = DEFAULT_BUCKET
) -> list[str]:
    try:
        s3_client = get_s3_client()
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        object_keys = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if not suffix or key.endswith(suffix):
                        object_keys.append(key)
        
        return object_keys
    except Exception as e:
        logger.error(f"Error listing objects in {bucket}/{prefix}: {str(e)}")
        raise
    
def read_csv_from_s3(
    key: str,
    bucket: str = DEFAULT_BUCKET,
    **kwargs
) -> pd.DataFrame:
    try:
        if not check_file_exists(key, bucket):
            logger.error(f"File s3://{bucket}/{key} does not exist")
            raise FileNotFoundError(f"File s3://{bucket}/{key} does not exist")
            
        s3_client = get_s3_client()
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        
        logger.info(f"Reading CSV file from s3://{bucket}/{key}")
        
        return pd.read_csv(obj['Body'], **kwargs)
    except ClientError as e:
        logger.error(f"S3 error reading CSV: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        raise
    
def read_parquet_from_s3(
    key: str,
    bucket: str = DEFAULT_BUCKET,
    columns: list[str] = None, 
    filters = None
) -> pd.DataFrame:
    try:
        if not check_file_exists(key, bucket):
            logger.error(f"File s3://{bucket}/{key} does not exist")
            raise FileNotFoundError(f"File s3://{bucket}/{key} does not exist")
            
        s3_client = get_s3_client()
        
        logger.info(f"Reading Parquet file from s3://{bucket}/{key}")
        
        buffer = s3_client.get_object(Bucket=bucket, Key=key)['Body'].read()
        table = pq.read_table(pa.py_buffer(buffer), columns=columns, filters=filters)
        return table.to_pandas()
    except ClientError as e:
        logger.error(f"S3 error reading Parquet: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error reading Parquet: {str(e)}")
        raise

def read_json_from_s3(
    key: str,
    bucket: str = DEFAULT_BUCKET,
    orient: str = 'records',
    lines: bool = False,
    **kwargs
) -> pd.DataFrame:
    try:
        if not check_file_exists(key, bucket):
            logger.error(f"File s3://{bucket}/{key} does not exist")
            raise FileNotFoundError(f"File s3://{bucket}/{key} does not exist")
            
        s3_client = get_s3_client()
        
        logger.info(f"Reading JSON file from s3://{bucket}/{key}")
        
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        
        try:
            return pd.read_json(obj['Body'], orient=orient, lines=lines, **kwargs)
        except:
            return json.loads(obj['Body'].read().decode('utf-8'))
    except ClientError as e:
        logger.error(f"S3 error reading JSON: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error reading JSON: {str(e)}")
        raise

def prepare_metadata(
    custom_metadata: dict[str, str] = None, 
    file_type: str = '', 
    shape: tuple[int, int] = None
) -> dict[str, str]:
    timestamp = datetime.now().isoformat()
    
    metadata = {
        'created-at': timestamp,
        'created-by': os.environ.get('USER', 'unknown')
    }
    
    if file_type:
        metadata['file-type'] = file_type
    
    if shape:
        metadata['rows'] = str(shape[0])
        metadata['columns'] = str(shape[1])
    
    if custom_metadata:
        sanitized_metadata = {k.lower(): str(v) for k, v in custom_metadata.items()}
        metadata.update(sanitized_metadata)
    
    return metadata

def write_csv_to_s3(
    df: pd.DataFrame,
    key: str, 
    bucket: str = DEFAULT_BUCKET,
    metadata: dict[str, str] = None,
    **kwargs
) -> None:
    try:
        s3_client = get_s3_client()
        
        logger.info(f"Writing CSV file to s3://{bucket}/{key}")
        
        csv_args = {
            'index': False, 
            'encoding': 'utf-8'
        }
        csv_args.update(kwargs)
        
        csv_buffer = df.to_csv(**csv_args)
        
        s3_metadata = prepare_metadata(metadata, 'csv', df.shape)
        
        s3_client.put_object(
            Body=csv_buffer,
            Bucket=bucket,
            Key=key,
            ContentType='text/csv',
            Metadata=s3_metadata
        )
    except Exception as e:
        logger.error(f"Error writing CSV to S3: {str(e)}")
        raise

def write_parquet_to_s3(
    df: pd.DataFrame,
    key: str, 
    bucket: str = DEFAULT_BUCKET,
    partition_cols: list[str] = None,
    metadata: dict[str, str] = None, 
    **kwargs
) -> None:
    try:
        s3_client = get_s3_client()
        
        logger.info(f"Writing Parquet file to s3://{bucket}/{key}")
        
        s3_metadata = prepare_metadata(metadata, 'parquet', df.shape)
        
        if partition_cols:
            table = pa.Table.from_pandas(df)
            
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp()
            local_path = os.path.join(temp_dir, "temp_partitioned")
            
            try:
                pq.write_to_dataset(
                    table, 
                    local_path, 
                    partition_cols=partition_cols,
                    **kwargs
                )
                
                for root, _, files in os.walk(local_path):
                    for file in files:
                        local_file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_file_path, local_path)
                        s3_file_key = os.path.join(key, relative_path)
                        
                        with open(local_file_path, 'rb') as f:
                            s3_client.put_object(
                                Body=f,
                                Bucket=bucket,
                                Key=s3_file_key,
                                ContentType='application/octet-stream',
                                Metadata=s3_metadata
                            )
            finally:
                shutil.rmtree(temp_dir)
        else:
            buffer = pa.BufferOutputStream()
            table = pa.Table.from_pandas(df)
            pq.write_table(table, buffer, **kwargs)
            
            s3_client.put_object(
                Body=buffer.getvalue().to_pybytes(),
                Bucket=bucket,
                Key=key,
                ContentType='application/octet-stream',
                Metadata=s3_metadata
            )
    except Exception as e:
        logger.error(f"Error writing Parquet to S3: {str(e)}")
        raise

def write_json_to_s3(
    data: pd.DataFrame,
    key: str, 
    bucket: str = DEFAULT_BUCKET,
    orient: str = 'records', 
    lines: bool = False,
    metadata: dict[str, str] = None, 
    **kwargs
) -> None:
    try:
        s3_client = get_s3_client()
        
        logger.info(f"Writing JSON file to s3://{bucket}/{key}")
        
        if isinstance(data, pd.DataFrame):
            json_data = data.to_json(orient=orient, lines=lines, **kwargs)
            shape = data.shape
        else:
            json_data = json.dumps(data, **kwargs)
            if isinstance(data, list):
                shape = (len(data), 1)
            elif isinstance(data, dict):
                shape = (1, len(data))
            else:
                shape = (1, 1)
        
        s3_metadata = prepare_metadata(metadata, 'json', shape)
        
        s3_client.put_object(
            Body=json_data,
            Bucket=bucket,
            Key=key,
            ContentType='application/json',
            Metadata=s3_metadata
        )
    except Exception as e:
        logger.error(f"Error writing JSON to S3: {str(e)}")
        raise

def get_object_metadata(
    key: str,
    bucket: str = DEFAULT_BUCKET
) -> dict[str, str]:
    try:
        s3_client = get_s3_client()
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return response.get('Metadata', {})
    except ClientError as e:
        logger.error(f"Error getting metadata: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting metadata: {str(e)}")
        raise

def copy_object(
    source_key: str, 
    dest_key: str,
    source_bucket: str = DEFAULT_BUCKET,
    dest_bucket: str = DEFAULT_BUCKET,
    metadata: dict[str, str] = None
) -> None:
    try:
        s3_client = get_s3_client()
        
        logger.info(f"Copying s3://{source_bucket}/{source_key} to s3://{dest_bucket}/{dest_key}")
        
        if metadata is None:
            response = s3_client.head_object(Bucket=source_bucket, Key=source_key)
            metadata = response.get('Metadata', {})
        
        metadata['copied-at'] = datetime.now().isoformat()
        
        s3_client.copy_object(
            CopySource={'Bucket': source_bucket, 'Key': source_key},
            Bucket=dest_bucket,
            Key=dest_key,
            Metadata=metadata,
            MetadataDirective='REPLACE'
        )
    except Exception as e:
        logger.error(f"Error copying object: {str(e)}")
        raise