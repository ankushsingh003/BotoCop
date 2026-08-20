import os

# MinIO speaks the S3 API, so boto3 works against it unmodified -- just
# point endpoint_url at the MinIO server instead of AWS. Swapping this
# for real AWS S3 later is a one-line config change, not a code change.
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
DATALAKE_BUCKET = os.getenv("DATALAKE_BUCKET", "botocop-datalake")
