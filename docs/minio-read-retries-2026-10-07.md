> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Complete MinIO reads with three delayed retries

The continuation failed on 6 October while rank 1 fetched an image range from
`dataset/raw/data/shard_064.tar`. The connection was refused; torchrun then stopped
the other ranks. This patch changes data access, not the model or training objective.

- Initial GET plus three retries, delayed by 5, 10 and 20 seconds.
- Retry the complete GET/body read, with exactly the same bucket, object and range.
- Retry transient network errors, incomplete transfers, HTTP 408/429/5xx and selected
  throttling/timeout errors. Authentication errors, missing objects and programming
  errors propagate immediately. An exhausted transient failure also propagates.
- Close every response, discard partial data, keep image and manifest SHA-256 checks.
- Advance an owned sample's loader cursor only after fetching and encoding succeeds.
  Ordered prefetch continues to save the consumer cursor, not the producer cursor.
- Apply the same policy to full-stream manifests/images and periodic validation.
- Checkpoint read-back verification uses a streaming SHA-256 calculation that starts
  over after a transient read failure. Checkpoint upload SDK retries remain unchanged.
- Use a dedicated reading client with `total_max_attempts=1`, connect timeout 10 s
  and read timeout 30 s. The application owns the four attempts; nested SDK retries
  cannot silently multiply them. Distributed collective timeouts remain unchanged.
- Log `MINIO_READ_RETRY`, `MINIO_READ_RECOVERED` and `MINIO_READ_EXHAUSTED`, including
  rank, object, attempt and delay. Never log credentials or raw exception URLs.

CPU tests cover actual boto3 HTTP 503 recovery, three connection errors followed by
success, body-read timeout, partial transfer, retry exhaustion, error classification,
checkpoint hashing, four-rank coverage and exact data-cursor replay. They do not prove
a new GPU training update has completed. The failed historical job is not restarted
by this patch; a continuation must restore the verified latest checkpoint and use the
corrected code rather than rerunning the previous parent checkpoint configuration.

References: [Boto3 retry configuration](https://docs.aws.amazon.com/boto3/latest/guide/retries.html)
and [StreamingBody reads](https://docs.aws.amazon.com/botocore/latest/reference/response.html).

Run targeted CPU tests with the project scripts on PYTHONPATH:

```sh
PYTHONPATH=scripts python -m unittest discover -s tests -p 'test_minio_retry.py' -v
PYTHONPATH=scripts python -m unittest discover -s tests -p 'test_fast_stream_contract.py' -v
PYTHONPATH=scripts python -m unittest discover -s tests -p 'test_distributed_loader.py' -v
PYTHONPATH=scripts python -m unittest discover -s tests -p 'test_ordered_prefetch.py' -v
```
