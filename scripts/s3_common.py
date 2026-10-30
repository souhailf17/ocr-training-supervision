"""Environment-only credentials; never log credential values."""
import os
import boto3
from botocore.config import Config

PROJECT = 'q35ocr-20260921-p560'
BUCKET = os.getenv('PROJECT_BUCKET', 'qwen35-9b-ocr-finetuning')

def client(*, streaming=False):
    # A complete-read wrapper owns retries for data and checkpoint reads.
    # Avoid nested retries; keep existing SDK retries for uploads/other calls.
    retries = {'total_max_attempts': 1, 'mode': 'standard'} if streaming else {'max_attempts': 5, 'mode': 'standard'}
    return boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'],
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        config=Config(signature_version='s3v4', connect_timeout=10 if streaming else 15,
                      read_timeout=30 if streaming else 120,
                      retries=retries, s3={'addressing_style': 'path'}))
