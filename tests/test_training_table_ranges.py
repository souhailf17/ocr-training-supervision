import io
import json
import tarfile
import unittest
from inspect_training_tables import annotation_range, classify, header_matches, range_bytes


class FakeS3:
    def __init__(self, data, full_response=False):
        self.data=data;self.full_response=full_response
    def get_object(self, Bucket, Key, Range):
        first,last=map(int,Range[6:].split('-'))
        if self.full_response:return {'Body':io.BytesIO(self.data)}
        return {'Body':io.BytesIO(self.data[first:last+1]),'ContentRange':f'bytes {first}-{last}/{len(self.data)}'}


def archive(order):
    annotation=json.dumps({'markdown':'[tbl-0.html](tbl-0.html)','tables':[]}).encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as stream:
        for name in order:
            data=annotation if name.endswith('.json') else b'PNG'*1400
            info=tarfile.TarInfo(name);info.size=len(data)
            stream.addfile(info,io.BytesIO(data))
    data=buffer.getvalue()
    with tarfile.open(fileobj=io.BytesIO(data)) as stream:
        image=stream.getmember('sample_1.png')
        row={'id':'data/shard_001.tar:sample_1','annotation_member':'sample_1.json',
             'image':{'key':'dataset/raw/data/shard_001.tar','offset':image.offset_data,'size':image.size}}
    return data,annotation,row


class TrainingTableRangeTests(unittest.TestCase):
    def test_json_before_image(self):
        data,expected,row=archive(['sample_1.json','sample_1.png'])
        found,header=annotation_range(FakeS3(data),row,len(data))
        self.assertEqual(found,expected);self.assertEqual(header,0)
    def test_json_after_image(self):
        data,expected,row=archive(['sample_1.png','sample_1.json'])
        found,header=annotation_range(FakeS3(data),row,len(data))
        self.assertEqual(found,expected);self.assertGreater(header,row['image']['offset'])
    def test_wrong_member_refused(self):
        data,_,row=archive(['sample_1.png','sample_1.json'])
        row['annotation_member']='different.json'
        with self.assertRaises(ValueError):annotation_range(FakeS3(data),row,len(data))
    def test_bad_header_checksum_refused(self):
        data,_,_=archive(['sample_1.json','sample_1.png'])
        header=bytearray(data[:512]);header[1]^=1
        self.assertIsNone(header_matches(bytes(header),'sample_1.json'))
    def test_server_ignoring_range_refused(self):
        data,_,_=archive(['sample_1.json','sample_1.png'])
        with self.assertRaises(ValueError):range_bytes(FakeS3(data,True),'key',512,10,len(data))
    def test_detection_includes_links_not_just_pipes(self):
        self.assertEqual(classify('[tbl-0.html](tbl-0.html)')['table_links'],1)
        self.assertFalse(classify('a|b')['pipe_heuristic'])
        self.assertTrue(classify('|a|b|\n|---|---|\n|c|d|')['pipe_heuristic'])


if __name__=='__main__':unittest.main()
