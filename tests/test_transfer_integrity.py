"""Offline corruption checks for the S3 read-back verification boundary."""
import hashlib, unittest
from sync_hf_to_s3 import verify

class Body:
    def __init__(self,data): self.data=data
    def iter_chunks(self,chunk_size): yield self.data
    def close(self): pass

class S3:
    def __init__(self,data,size=None): self.data=data; self.size=len(data) if size is None else size
    def head_object(self,**kwargs): return {'ContentLength':self.size}
    def get_object(self,**kwargs): return {'Body':Body(self.data)}

class IntegrityTests(unittest.TestCase):
    def test_correct_object(self):
        verify(S3(b'original'),'object',8,hashlib.sha256(b'original').hexdigest())
    def test_equal_size_corruption_rejected(self):
        with self.assertRaises(ValueError):
            verify(S3(b'corrupt!'),'object',8,hashlib.sha256(b'original').hexdigest())
    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            verify(S3(b'bad'),'object',8,hashlib.sha256(b'original').hexdigest())
    def test_truncated_get_despite_correct_head_rejected(self):
        with self.assertRaises(ValueError):
            verify(S3(b'orig',size=8),'object',8,hashlib.sha256(b'original').hexdigest())

if __name__=='__main__': unittest.main()
