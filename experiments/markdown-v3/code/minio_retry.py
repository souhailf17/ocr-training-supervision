"""Retry complete, idempotent S3 reads without advancing the data cursor.

Use s3_common.client(streaming=True): SDK retries are disabled on that client
so the initial request plus three application retries cannot multiply silently.
Normal uploads keep their existing SDK/transfer retry policy.
"""
import errno
import hashlib
import json
import os
import re
import time

from botocore.exceptions import (
    ClientError, ConnectionClosedError, ConnectTimeoutError,
    EndpointConnectionError, IncompleteReadError, ReadTimeoutError,
    ResponseStreamingError,
)

RETRY_DELAYS = (5, 10, 20)
NETWORK_ERRORS = (
    EndpointConnectionError, ConnectionClosedError, ConnectTimeoutError,
    ReadTimeoutError, ResponseStreamingError, IncompleteReadError,
)
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
TRANSIENT_CODES = {
    'RequestTimeout', 'RequestTimeoutException', 'SlowDown', 'Throttling',
    'ThrottlingException', 'InternalError', 'InternalFailure',
    'ServiceUnavailable',
}


def retryable(exc):
    if isinstance(exc, NETWORK_ERRORS):
        return True
    if isinstance(exc, ClientError):
        response = exc.response
        return (response.get('ResponseMetadata', {}).get('HTTPStatusCode') in TRANSIENT_STATUS
                or response.get('Error', {}).get('Code') in TRANSIENT_CODES)
    return isinstance(exc, OSError) and exc.errno in {
        errno.ECONNREFUSED, errno.ECONNRESET, errno.ECONNABORTED,
        errno.ETIMEDOUT, errno.EPIPE, errno.ENETUNREACH, errno.EHOSTUNREACH,
    }


def _log(event, key, attempt, **fields):
    # Never print exception text, signed URLs, headers or credentials.
    print('MINIO_READ_' + event, json.dumps({
        'timestamp': time.time(), 'rank': os.getenv('RANK', 'unknown'),
        'key': key, 'attempt': attempt, 'max_attempts': 4, **fields,
    }), flush=True)


def _read(s3, bucket, key, consume, byte_range=None):
    request = {'Bucket': bucket, 'Key': key}
    if byte_range is not None:
        request['Range'] = byte_range
    for attempt in range(1, 5):
        try:
            response = s3.get_object(**request)
            body = response['Body']
            try:
                result = consume(body, response)
            finally:
                body.close()
        except Exception as exc:
            if not retryable(exc):
                raise
            if attempt == 4:
                _log('EXHAUSTED', key, attempt, error_type=type(exc).__name__)
                raise
            delay = RETRY_DELAYS[attempt - 1]
            _log('RETRY', key, attempt, retry_in_seconds=delay,
                 error_type=type(exc).__name__, byte_range=byte_range)
            time.sleep(delay)
        else:
            if attempt > 1:
                _log('RECOVERED', key, attempt, byte_range=byte_range)
            return result


def read_object(s3, bucket, key, byte_range=None):
    """Reopen and reread the same object/range after a transient failure.

    Discard partial bytes, close every response, and return only a complete
    read. Existing image/manifest SHA checks remain at the caller.
    """
    expected_range_size = None
    if byte_range is not None:
        match = re.fullmatch(r'bytes=(\d+)-(\d+)', byte_range)
        if not match or int(match[2]) < int(match[1]):
            raise ValueError('Expected a closed, nonempty byte range')
        expected_range_size = int(match[2]) - int(match[1]) + 1

    def consume(body, response):
        raw = body.read()
        expected = response.get('ContentLength')
        if expected is not None and len(raw) != expected:
            raise IncompleteReadError(actual_bytes=len(raw), expected_bytes=expected)
        if expected_range_size is not None and len(raw) != expected_range_size:
            raise IncompleteReadError(actual_bytes=len(raw), expected_bytes=expected_range_size)
        return raw

    return _read(s3, bucket, key, consume, byte_range)


def object_sha256(s3, bucket, key):
    """Verify large checkpoints without buffering them; reset hash on retry."""
    def consume(body, response):
        digest = hashlib.sha256()
        size = 0
        for chunk in body.iter_chunks(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        expected = response.get('ContentLength')
        if expected is not None and size != expected:
            raise IncompleteReadError(actual_bytes=size, expected_bytes=expected)
        return digest.hexdigest()

    return _read(s3, bucket, key, consume)
