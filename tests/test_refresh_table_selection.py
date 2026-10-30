import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from annotation_targets import TARGET_FORMAT,resolve_annotation
from refresh_table_selection import refresh_selection
from test_annotation_targets import annotation

class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_root=(Path(__file__).parent/'.tmp').resolve()
        cls.temporary_root.mkdir(exist_ok=True)
    def test_same_page_order_image_hash_and_split_are_preserved(self):
        out=resolve_annotation(annotation())
        old={'id':'page','group_id':'g','split':'validation','image':{'sha256':'same-image'},'text':'old reference'}
        new=dict(old,text=out.text,text_sha256=hashlib.sha256(out.text.encode()).hexdigest(),target_format=TARGET_FORMAT)
        raw=(json.dumps(new)+'\n').encode()
        class S3:
            def get_object(self,**kwargs):return {'Body':io.BytesIO(raw)}
        summary={'target_format':TARGET_FORMAT,'shards':[{'manifest':{'key':'v2','sha256':hashlib.sha256(raw).hexdigest()}}]}
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as t:
            self.assertTrue(Path(t).resolve().is_relative_to(self.temporary_root))
            src,dst=Path(t)/'old.jsonl',Path(t)/'new.jsonl';src.write_text(json.dumps(old)+'\n')
            before=src.read_bytes()
            self.assertEqual(refresh_selection(S3(),'bucket',summary,src,dst)['pages'],1)
            self.assertEqual(src.read_bytes(),before)
            self.assertEqual(json.loads(dst.read_text())['image']['sha256'],'same-image')
    def test_changed_split_is_refused_before_output_is_written(self):
        old={'id':'page','group_id':'g','split':'validation','image':{'sha256':'same-image'},'text':'old'}
        new=dict(old,split='train',target_format=TARGET_FORMAT)
        raw=(json.dumps(new)+'\n').encode()
        class S3:
            def get_object(self,**kwargs):return {'Body':io.BytesIO(raw)}
        summary={'target_format':TARGET_FORMAT,'shards':[{'manifest':{'key':'v2','sha256':hashlib.sha256(raw).hexdigest()}}]}
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as t:
            self.assertTrue(Path(t).resolve().is_relative_to(self.temporary_root))
            src,dst=Path(t)/'old.jsonl',Path(t)/'new.jsonl';src.write_text(json.dumps(old)+'\n')
            with self.assertRaisesRegex(ValueError,'identity/split'):refresh_selection(S3(),'bucket',summary,src,dst)
            self.assertFalse(dst.exists())

if __name__=='__main__':unittest.main()
