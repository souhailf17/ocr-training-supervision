"""Exercise the actual sharding/resume class without CUDA or Arsenal imports."""
import ast
import hashlib
import io
import json
import random
import unittest
from pathlib import Path
from minio_retry import read_object


ROOT = Path(__file__).parent
SCRIPT_ROOT = ROOT.parent / "scripts"


def class_source(filename, name):
    text = (SCRIPT_ROOT / filename).read_text()
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == name)
    return ast.get_source_segment(text, node)


class FakeDist:
    def __init__(self, rank, world):
        self.rank = rank
        self.world = world

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world


ROWS = [{"id": index, "split": "train", "image": {"key": "image", "offset": 0, "size": 1}} for index in range(24)]
RAW = b"\n".join(json.dumps(row).encode() for row in ROWS)


class FakeS3:
    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(RAW if kwargs["Key"] == "manifest" else b"x")}


def loader(state, rank, world):
    namespace = {
        "OCRLoader": object,
        "dist": FakeDist(rank, world),
        "hashlib": hashlib,
        "json": json,
        "random": random,
        "BUCKET": "test",
        "read_object": read_object,
    }
    exec(class_source("train_full.py", "StreamingLoader"), namespace)
    exec(class_source("distributed_ocr.py", "RankStream"), namespace)
    stream = namespace["RankStream"].__new__(namespace["RankStream"])
    stream.reports = [{"manifest": {"key": "manifest", "sha256": hashlib.sha256(RAW).hexdigest()}}]
    stream.s3 = FakeS3()
    stream.encode = lambda row, raw: row["id"]
    stream.load_state_dict(state)
    return stream


class DistributedLoaderTests(unittest.TestCase):
    def test_reshard_from_four_to_two_ranks_without_repeating_examples(self):
        expected = ROWS.copy()
        random.Random(42).shuffle(expected)
        expected = [row["id"] for row in expected]
        state = {"shard": 0, "cursor": 8, "count": 8}
        for start, world in ((8, 4), (16, 2)):
            seen = []
            states = []
            for rank in range(world):
                stream = loader(state, rank, world)
                iterator = iter(stream)
                seen.extend(next(iterator) for _ in range(8 // world))
                states.append(stream.state_dict())
            self.assertEqual(len(set(seen)), 8)
            self.assertEqual(set(seen), set(expected[start : start + 8]))
            state = max(states, key=lambda value: value["count"])
            self.assertEqual(state["count"], start + 8)


if __name__ == "__main__":
    unittest.main()
