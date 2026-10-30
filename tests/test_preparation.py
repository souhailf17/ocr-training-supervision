"""Synthetic fixtures test label preservation and leakage protection, not model training."""
import io,json,sqlite3,unittest
from project_runtime import ROOT
from PIL import Image
from prepare_dataset import pair_record,choose_split,group_id

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:')
        self.db.execute('CREATE TABLE documents (id TEXT PRIMARY KEY,text_hash TEXT,split TEXT)')
        self.db.execute('CREATE TABLE images (hash TEXT PRIMARY KEY,text_hash TEXT,split TEXT)')
        self.db.execute('CREATE TABLE rejected_documents (id TEXT PRIMARY KEY,reason TEXT)')
        out=io.BytesIO(); Image.new('RGB',(4,4),'white').save(out,format='PNG'); self.image=out.getvalue()
    def tearDown(self): self.db.close()
    def pair(self,text,aug=False):
        annotation={'markdown':text,'meta':{'augmentation':{'name':'test'} if aug else None}}
        return {'json':{'name':'sample.json','data':json.dumps(annotation).encode()},'image':{'name':'sample.png','data':self.image,'offset':512}}
    def test_original_and_augmentation_match(self):
        text='النص العربي\n$E=mc^2$'
        original=pair_record(self.db,'data/shard_001.tar','sample_0000000',self.pair(text))
        aug=pair_record(self.db,'data_aug/shard_001_aug1.tar','sample_0000000_01',self.pair(text,True))
        self.assertEqual(original['split'],aug['split']); self.assertEqual(original['group_id'],aug['group_id'])
        self.assertEqual(original['text'],text)
    def test_changed_augmentation_target_fails(self):
        pair_record(self.db,'data/shard_001.tar','sample_0000000',self.pair('original'))
        with self.assertRaisesRegex(ValueError,'differs from original'):
            pair_record(self.db,'data_aug/shard_001_aug1.tar','sample_0000000_01',self.pair('wrong',True))
    def test_duplicate_image_conflicting_label_fails(self):
        pair_record(self.db,'data/shard_001.tar','sample_0000000',self.pair('label one'))
        with self.assertRaisesRegex(ValueError,'conflicting label'):
            pair_record(self.db,'data/shard_002.tar','sample_0000001',self.pair('label two'))
    def test_equal_normalized_targets_share_split(self):
        self.assertEqual(choose_split('example\r\ntext'),choose_split('example\ntext'))
    def test_unknown_naming_fails(self):
        with self.assertRaises(ValueError): group_id('data_aug/shard_001_aug2.tar','sample_0000000_01')
    def test_empty_original_excludes_augmented_group(self):
        original=pair_record(self.db,'data/shard_001.tar','sample_0000000',self.pair(''))
        aug=pair_record(self.db,'data_aug/shard_001_aug1.tar','sample_0000000_01',self.pair('',True))
        self.assertIn('rejected',original); self.assertIn('rejected',aug)
    def test_zero_filled_json_excludes_group(self):
        bad=self.pair('unused'); bad['json']['data']=bytes(1772)
        original=pair_record(self.db,'data/shard_007.tar','sample_0066435',bad)
        aug=pair_record(self.db,'data_aug/shard_007_aug1.tar','sample_0066435_01',self.pair('untrusted',True))
        self.assertEqual(original['rejected'],'malformed_json_annotation')
        self.assertIn('source_document_rejected',aug['rejected'])
        self.assertIn('annotation_sha256',original)
    def test_truncated_utf32_json_excluded(self):
        bad=self.pair('unused'); bad['json']['data']=bytes(21061)
        self.assertEqual(pair_record(self.db,'data/shard_007.tar','sample_0066436',bad)['rejected'],'malformed_json_annotation')
    def test_nonobject_json_excluded(self):
        bad=self.pair('unused'); bad['json']['data']=b'[]'
        self.assertEqual(pair_record(self.db,'data/shard_001.tar','sample_0000000',bad)['rejected'],'annotation_is_not_an_object')
    def test_unreadable_image_excludes_original_group(self):
        bad=self.pair('real label'); bad['image']['data']=bytes(128)
        original=pair_record(self.db,'data/shard_001.tar','sample_0000000',bad)
        aug=pair_record(self.db,'data_aug/shard_001_aug1.tar','sample_0000000_01',self.pair('real label',True))
        self.assertEqual(original['rejected'],'undecodable_image'); self.assertIn('rejected',aug)

if __name__=='__main__': unittest.main()
