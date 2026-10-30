"""CPU test of actual sharding/prefetch across shard and checkpoint boundaries.

Extract loader class bodies so testing needs neither CUDA nor Arsenal installed.
"""
import ast
import hashlib
import io
import json
import random
import contextlib
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from ordered_prefetch import OrderedPrefetch
from minio_retry import read_object
from botocore.exceptions import EndpointConnectionError

ROOT = Path(__file__).parent
SCRIPT_ROOT = ROOT.parent / "scripts"


def class_source(filename, name):
    text = (SCRIPT_ROOT/filename).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)
    return ast.get_source_segment(text, node)


class Rank:
    def __init__(self, rank, world=4): self.rank = rank; self.world = world
    def get_world_size(self): return self.world
    def get_rank(self): return self.rank


def stream(rank, reports, objects, state, world=4):
    ns = {'OCRLoader': object, 'dist': Rank(rank, world), 'hashlib': hashlib, 'json': json, 'random': random, 'BUCKET': 'test', 'read_object': read_object}
    exec(class_source('train_full.py', 'StreamingLoader'), ns)
    exec(class_source('distributed_ocr.py', 'RankStream'), ns)
    cls = ns['RankStream']
    obj = cls.__new__(cls)
    obj.reports = reports
    class S3:
        def get_object(self, **kwargs): return {'Body': io.BytesIO(objects[kwargs['Key']])}
    obj.s3 = S3()
    obj.encode = lambda row, raw: row['id']
    obj.load_state_dict(state)
    return OrderedPrefetch(obj, depth=4)


class ContractTest(unittest.TestCase):
    def test_retry_preserves_all_rank_coverage_and_checkpoint_replay(self):
        rows = [{'id': i, 'split': 'train', 'image': {'key': 'img', 'offset': 0, 'size': 1}} for i in range(40)]
        raw = b'\n'.join(json.dumps(row).encode() for row in rows)
        reports = [{'manifest': {'key': 'manifest', 'sha256': hashlib.sha256(raw).hexdigest()}}]
        objects = {'manifest': raw, 'img': b'x'}
        def make_loaders():
            loaders = [stream(rank, reports, objects, {'shard': 0, 'cursor': 0, 'count': 0}) for rank in range(4)]
            for loader in loaders:
                s3 = loader.inner.s3
                original = s3.get_object
                s3.failed_reads = 0
                def flaky(_s3=s3, _original=original, **kwargs):
                    if kwargs['Key'] == 'img' and _s3.failed_reads < 3:
                        _s3.failed_reads += 1
                        raise EndpointConnectionError(endpoint_url='http://unavailable.invalid')
                    return _original(**kwargs)
                s3.get_object = flaky
            return loaders
        with patch('minio_retry.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            loaders = make_loaders()
            iterators = [iter(loader) for loader in loaders]
            before = [[next(it) for _ in range(4)] for it in iterators]
            canonical = max((loader.state_dict() for loader in loaders), key=lambda s: s['count'])
            self.assertEqual(canonical['count'], 16)
            expected_next = [[next(it) for _ in range(2)] for it in iterators]
            for it in iterators:
                it.close()
            for loader in loaders:
                loader.load_state_dict(canonical)
            iterators = [iter(loader) for loader in loaders]
            replay = [[next(it) for _ in range(2)] for it in iterators]
            self.assertEqual(replay, expected_next)
            tails = [list(it) for it in iterators]
        seen = sum((a+b+c for a,b,c in zip(before, replay, tails)), [])
        self.assertEqual(sorted(seen), list(range(40)))
        self.assertEqual(len(set(seen)), 40)
        self.assertTrue(all(loader.inner.s3.failed_reads == 3 for loader in loaders))

    def test_exhausted_read_does_not_commit_owned_example(self):
        rows = [{'id': i, 'split': 'train', 'image': {'key': 'img', 'offset': 0, 'size': 1}} for i in range(8)]
        raw = b'\n'.join(json.dumps(row).encode() for row in rows)
        reports = [{'manifest': {'key': 'manifest', 'sha256': hashlib.sha256(raw).hexdigest()}}]
        loader = stream(1, reports, {'manifest': raw, 'img': b'x'}, {'shard': 0, 'cursor': 0, 'count': 0})
        original = loader.inner.s3.get_object
        calls = []
        def broken(**kwargs):
            if kwargs['Key'] == 'img':
                calls.append(kwargs)
                raise EndpointConnectionError(endpoint_url='http://unavailable.invalid')
            return original(**kwargs)
        loader.inner.s3.get_object = broken
        with patch('minio_retry.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(EndpointConnectionError):
                next(iter(loader))
        self.assertEqual(len(calls), 4)
        self.assertEqual(loader.count, 0)  # Consumer has not received an example.
        self.assertEqual(loader.inner.count, 1)  # Only rank-0's skipped row advanced.
        self.assertEqual(loader.inner.cursor, 1)

    def test_four_to_two_rank_cursor_reshard(self):
        rows = [{'id': i, 'split': 'train', 'image': {'key': 'img', 'offset': 0, 'size': 1}} for i in range(300)]
        raw = b'\n'.join(json.dumps(row).encode() for row in rows)
        reports = [{'manifest': {'key': 'manifest', 'sha256': hashlib.sha256(raw).hexdigest()}}]
        objects = {'manifest': raw, 'img': b'x'}
        state = {'shard': 0, 'cursor': 0, 'count': 162048}
        loaders = [stream(rank, reports, objects, state, world=2) for rank in range(2)]
        iters = [iter(loader) for loader in loaders]
        for iterator in iters:
            for _ in range(128): next(iterator)
        states = [loader.state_dict() for loader in loaders]
        self.assertEqual([item['count'] for item in states], [162303, 162304])
        self.assertEqual(max(item['count'] for item in states), 162304)
        for iterator in iters: iterator.close()

    def test_disjoint_full_coverage_and_resume_with_lookahead(self):
        reports, objects, expected = [], {}, []
        offset = 0
        for shard, size in enumerate([13, 17, 10]):
            rows = [{'id': offset+i, 'split': 'train', 'image': {'key': 'img', 'offset': 0, 'size': 1}} for i in range(size)]
            offset += size
            raw = b'\n'.join(json.dumps(row).encode() for row in rows)
            key = str(shard)
            reports.append({'manifest': {'key': key, 'sha256': hashlib.sha256(raw).hexdigest()}})
            objects[key] = raw
            random.Random(42+shard).shuffle(rows)
            expected.extend(row['id'] for row in rows)
        objects['img'] = b'x'
        loaders = [stream(rank, reports, objects, {'shard': 0, 'cursor': 0, 'count': 0}) for rank in range(4)]
        iters = [iter(loader) for loader in loaders]
        before = [[next(it) for _ in range(4)] for it in iters]
        canonical = max((loader.state_dict() for loader in loaders), key=lambda state: state['count'])
        self.assertEqual(canonical['count'], 16)
        following = [[next(it) for _ in range(3)] for it in iters]
        for it in iters: it.close()
        for loader in loaders: loader.load_state_dict(canonical)
        iters = [iter(loader) for loader in loaders]
        replay = [[next(it) for _ in range(3)] for it in iters]
        self.assertEqual(following, replay)
        tails = [list(it) for it in iters]
        all_seen = sum((a+b+c for a,b,c in zip(before, replay, tails)), [])
        self.assertEqual(sorted(all_seen), sorted(expected))
        self.assertEqual(len(set(all_seen)), 40)
        for rank in range(4): self.assertEqual(before[rank]+replay[rank]+tails[rank], expected[rank::4])


if __name__ == '__main__': unittest.main()
