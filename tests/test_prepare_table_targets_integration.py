"""Exercise the actual pair_record source with SQLite/Pillow, without Linux/S3 imports."""
import ast
import hashlib
import io
import json
import re
import sqlite3
import unicodedata
import unittest
from pathlib import Path
from PIL import Image
from annotation_targets import MANIFEST_VERSION,resolve_annotation
from test_annotation_targets import annotation

def preparation_functions():
    path=Path(__file__).parents[1]/'scripts/prepare_dataset.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'sha','choose_split','group_id','pair_record'}]
    env={'hashlib':hashlib,'re':re,'json':json,'io':io,'Image':Image,'unicodedata':unicodedata,
         'SEED':42,'BUCKET':'owned-test-bucket','VERSION':MANIFEST_VERSION,'TARGET_BUILDER':resolve_annotation}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),env)
    return env

class PreparationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env=preparation_functions()
        self.db=sqlite3.connect(':memory:')
        self.db.executescript('CREATE TABLE documents(id TEXT PRIMARY KEY,text_hash TEXT,split TEXT);CREATE TABLE images(hash TEXT PRIMARY KEY,text_hash TEXT,split TEXT);CREATE TABLE rejected_documents(id TEXT PRIMARY KEY,reason TEXT);')
        image=io.BytesIO();Image.new('RGB',(2,2),'white').save(image,format='PNG');self.image=image.getvalue()
    def tearDown(self):self.db.close()
    def pair(self,a):
        return {'json':{'name':'sample.json','data':json.dumps(a,ensure_ascii=False).encode()},
                'image':{'name':'sample.png','offset':512,'data':self.image}}
    def test_new_cells_change_the_label_without_moving_the_legacy_split(self):
        a=annotation();a['meta']={'augmentation':None,'language':'ar'}
        choose=self.env['choose_split']
        for index in range(1000):
            a['markdown']=f'Header {index}\n[tbl-0.html](tbl-0.html)'
            if choose(a['markdown'])!=choose(resolve_annotation(a).text):break
        else:self.fail('No diagnostic split-changing fixture found')
        record=self.env['pair_record'](self.db,'data/shard_001.tar','sample_0000001',self.pair(a))
        self.assertEqual(record['split'],choose(a['markdown']))
        self.assertNotEqual(record['split'],choose(record['text']))
        self.assertEqual(record['text'],resolve_annotation(a).text)
        self.assertEqual(record['image']['key'],'dataset/raw/data/shard_001.tar')
        self.assertEqual(record['schema'],MANIFEST_VERSION)
        self.assertEqual(record['table_audit']['cell_count'],4)
    def test_original_and_augmentation_keep_the_same_target_and_split(self):
        a=annotation();a['meta']={'augmentation':None}
        first=self.env['pair_record'](self.db,'data/shard_001.tar','sample_0000001',self.pair(a))
        a['meta']['augmentation']='aug1'
        second=self.env['pair_record'](self.db,'data_aug/shard_001_aug1.tar','sample_0000001_01',self.pair(a))
        self.assertEqual(first['split'],second['split'])
        self.assertEqual(first['text_sha256'],second['text_sha256'])
        self.assertEqual(first['group_id'],second['group_id'])
    def test_legacy_preparation_keeps_its_original_target_and_schema(self):
        self.env['TARGET_BUILDER']=None;self.env['VERSION']='ocr-manifest-v1'
        a=annotation();a['meta']={'augmentation':None}
        record=self.env['pair_record'](self.db,'data/shard_001.tar','sample_0000001',self.pair(a))
        self.assertEqual(record['text'],a['markdown'])
        self.assertEqual(record['schema'],'ocr-manifest-v1')
        self.assertNotIn('table_audit',record)
    def test_incomplete_original_is_quarantined_with_its_augmentation(self):
        a=annotation();a['tables']=[];a['meta']={'augmentation':None}
        first=self.env['pair_record'](self.db,'data/shard_001.tar','sample_0000001',self.pair(a))
        self.assertEqual(first['rejected'],'invalid_table_target')
        a['meta']['augmentation']='aug1'
        second=self.env['pair_record'](self.db,'data_aug/shard_001_aug1.tar','sample_0000001_01',self.pair(a))
        self.assertTrue(second['rejected'].startswith('source_document_rejected'))

if __name__=='__main__':unittest.main()
