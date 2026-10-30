"""CPU-only transport failure, body-read, retry bound and client-policy tests."""
import contextlib
import hashlib
import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
import minio_retry as retry
from s3_common import client


class Body(io.BytesIO):
    def iter_chunks(self, size):
        while chunk := self.read(size):
            yield chunk


class ScriptedS3:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def get_object(self, **request):
        self.calls.append(request)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return {'Body': result, 'ContentLength': 3}


def refused():
    return EndpointConnectionError(endpoint_url='http://unavailable.invalid')


class RetryTests(unittest.TestCase):
    def test_three_retries_then_same_range_succeeds(self):
        body = Body(b'xyz')
        s3 = ScriptedS3([refused(), refused(), refused(), body])
        with patch.object(retry.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(retry.read_object(s3, 'bucket', 'image', 'bytes=10-12'), b'xyz')
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10, 20])
        self.assertEqual(s3.calls, [{'Bucket': 'bucket', 'Key': 'image', 'Range': 'bytes=10-12'}] * 4)
        self.assertTrue(body.closed)
        self.assertIn('MINIO_READ_RECOVERED', log.getvalue())
        self.assertNotIn('unavailable.invalid', log.getvalue())

    def test_persistent_failure_stops_after_four_requests(self):
        errors = [refused() for _ in range(4)]
        s3 = ScriptedS3(errors)
        with patch.object(retry.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(EndpointConnectionError) as caught:
                retry.read_object(s3, 'bucket', 'image')
        self.assertIs(caught.exception, errors[-1])
        self.assertEqual(len(s3.calls), 4)
        self.assertEqual(sleep.call_count, 3)

    def test_body_timeout_closes_response_and_restarts_read(self):
        class BrokenBody(Body):
            def read(self, *args):
                raise ReadTimeoutError(endpoint_url='http://hidden.invalid')
        broken, good = BrokenBody(b'x'), Body(b'xyz')
        s3 = ScriptedS3([broken, good])
        with patch.object(retry.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(retry.read_object(s3, 'bucket', 'image'), b'xyz')
        self.assertTrue(broken.closed and good.closed)
        self.assertEqual(len(s3.calls), 2)

    def test_partial_read_is_discarded(self):
        partial = Body(b'x')
        s3 = ScriptedS3([partial, Body(b'xyz')])
        with patch.object(retry.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(retry.read_object(s3, 'bucket', 'image', 'bytes=10-12'), b'xyz')
        self.assertTrue(partial.closed)

    def test_bad_range_is_rejected_before_network(self):
        s3 = ScriptedS3([])
        with self.assertRaises(ValueError):
            retry.read_object(s3, 'bucket', 'image', 'bytes=20-10')
        self.assertEqual(s3.calls, [])

    def test_nontransient_errors_are_not_retried(self):
        for code, status in [('AccessDenied', 403), ('NoSuchKey', 404)]:
            error = ClientError({'Error': {'Code': code}, 'ResponseMetadata': {'HTTPStatusCode': status}}, 'GetObject')
            s3 = ScriptedS3([error])
            with patch.object(retry.time, 'sleep') as sleep:
                with self.assertRaises(ClientError):
                    retry.read_object(s3, 'bucket', 'image')
            sleep.assert_not_called()
            self.assertEqual(len(s3.calls), 1)

    def test_programming_error_is_not_retried(self):
        with patch.object(retry.time, 'sleep') as sleep:
            with self.assertRaises(ValueError):
                retry.read_object(ScriptedS3([ValueError('invalid input')]), 'bucket', 'image')
        sleep.assert_not_called()

    def test_checkpoint_hash_resets_after_partial_stream_failure(self):
        class BrokenBody(Body):
            def iter_chunks(self, size):
                yield b'x'
                raise ReadTimeoutError(endpoint_url='http://hidden.invalid')
        broken = BrokenBody()
        s3 = ScriptedS3([broken, Body(b'xyz')])
        with patch.object(retry.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(retry.object_sha256(s3, 'bucket', 'checkpoint'), hashlib.sha256(b'xyz').hexdigest())
        self.assertTrue(broken.closed)

    def test_data_client_has_no_nested_sdk_retries(self):
        with patch.dict(os.environ, {'AWS_ENDPOINT_URL': 'http://test.invalid'}), patch('s3_common.boto3.client') as factory:
            client(streaming=True)
            config = factory.call_args.kwargs['config']
            self.assertEqual(config.retries['total_max_attempts'], 1)
            self.assertEqual((config.connect_timeout, config.read_timeout), (10, 30))
            client()
            self.assertEqual(factory.call_args.kwargs['config'].retries['max_attempts'], 5)

    def test_real_sdk_503_then_recovery_after_three_application_retries(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                requests.append((self.path, self.headers.get('Range')))
                if len(requests) <= 3:
                    payload = b'<Error><Code>ServiceUnavailable</Code></Error>'
                    self.send_response(503)
                else:
                    payload = b'xyz'
                    self.send_response(206)
                    self.send_header('Content-Range', 'bytes 10-12/20')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            s3 = boto3.client('s3', endpoint_url=f'http://127.0.0.1:{server.server_port}',
                              aws_access_key_id='test', aws_secret_access_key='test', region_name='us-east-1',
                              config=Config(retries={'total_max_attempts': 1}, s3={'addressing_style': 'path'}))
            with patch.object(retry.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(retry.read_object(s3, 'bucket', 'image', 'bytes=10-12'), b'xyz')
            self.assertEqual(requests, [('/bucket/image', 'bytes=10-12')] * 4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10, 20])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
