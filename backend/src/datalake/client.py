import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from backend.src.datalake.config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    DATALAKE_BUCKET,
)


def get_s3_client():
    kwargs = dict(
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(connect_timeout=1, read_timeout=1, retries={"max_attempts": 1}),
    )
    if MINIO_ENDPOINT:
        kwargs["endpoint_url"] = MINIO_ENDPOINT
    return boto3.client("s3", **kwargs)


def ensure_bucket(client, bucket: str = DATALAKE_BUCKET):
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
