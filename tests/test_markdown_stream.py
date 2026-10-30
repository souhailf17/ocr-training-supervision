import io,json,sys,unittest,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import markdown_stream as module
from markdown_stream import MarkdownStream,canonical_cursor
from markdown_table_targets import sha

class FakeS3:
    def __init__(self,objects):self.objects=objects
    def get_object(self,**kwargs):
        data=self.objects[kwargs['Key']];return {'Body':io.BytesIO(data),'ContentLength':len(data)}

def fixtures():
    reports=[];objects={};expected=[]
    for s,size in enumerate([9,11,8]):
        rows=[{'id':f'{s}:{n}','split':'validation' if n==0 else 'train','sequence_tokens':1,'supervised_tokens':1,'image':{'key':'image','offset':0,'size':1}} for n in range(size)]
        expected.extend(r['id'] for r in rows if r['split']=='train')
        raw=b'\n'.join(json.dumps(r).encode() for r in rows);key=f'm{s}';objects[key]=raw
        reports.append({'manifest':{'key':key,'sha256':sha(raw),'bytes':len(raw)},'counts':{'train':size-1}})
    objects['image']=b'x';return reports,FakeS3(objects),set(expected)

class StreamTests(unittest.TestCase):
    def setUp(self):
        self.old_read=module.read;self.old_encode=module.encode;self.old_image=MarkdownStream.read_image
        module.read=lambda s,k:s.objects[k]
        class Labels:
            def __getitem__(self,key):return self
            def __ne__(self,key):return self
            def sum(self):return 1
        module.encode=lambda p,r,raw,pin=False:{'_id':r['id'],'input_ids':types.SimpleNamespace(shape=(1,1)),'labels':Labels()}
        MarkdownStream.read_image=lambda self,image,start:b'x'
    def tearDown(self):module.read=self.old_read;module.encode=self.old_encode;MarkdownStream.read_image=self.old_image
    def test_exact_disjoint_coverage_and_no_validation(self):
        reports,s3,expected=fixtures();outputs=[]
        for rank in range(4):outputs.extend(b['_id'] for b in MarkdownStream(reports,s3,None,rank,4))
        self.assertEqual(set(outputs),expected);self.assertEqual(len(outputs),len(expected))
    def test_canonical_checkpoint_replay_cross_shards(self):
        reports,s3,expected=fixtures();streams=[MarkdownStream(reports,s3,None,r,4) for r in range(4)]
        it=[iter(s) for s in streams];before=[]
        for rank in range(4):before.extend(next(it[rank])['_id'] for _ in range(3))
        state=canonical_cursor([s.state_dict() for s in streams],12)
        after=[]
        for rank in range(4):
            stream=MarkdownStream(reports,s3,None,rank,4);stream.load_state_dict(state);after.extend(b['_id'] for b in stream)
        self.assertFalse(set(before)&set(after));self.assertEqual(set(before+after),expected)
        self.assertEqual(len(before+after),len(expected))
    def test_single_gpu_complete_pass_and_checkpoint_replay(self):
        reports,s3,expected=fixtures();stream=MarkdownStream(reports,s3,None,0,1)
        iterator=iter(stream);before=[next(iterator)['_id'] for _ in range(12)]
        state=canonical_cursor([stream.state_dict()],12)
        restored=MarkdownStream(reports,s3,None,0,1);restored.load_state_dict(state)
        after=[batch['_id'] for batch in restored]
        self.assertEqual(after,[batch['_id'] for batch in iterator])
        self.assertFalse(set(before)&set(after))
        self.assertEqual(set(before+after),expected)
        self.assertEqual(len(before+after),len(expected))
        self.assertEqual(canonical_cursor([restored.state_dict()],len(expected))['count'],len(expected))
    def test_failed_image_does_not_commit_owned_sample(self):
        reports,s3,_=fixtures();stream=MarkdownStream(reports,s3,None,0,4)
        module.encode=lambda *a,**k:(_ for _ in ()).throw(OSError('read failed'))
        with self.assertRaises(OSError):next(iter(stream))
        self.assertEqual(stream.count,0);self.assertEqual(stream.cursor,0)
    def test_changed_dataset_resume_rejected(self):
        reports,s3,_=fixtures();stream=MarkdownStream(reports,s3,None,0,4);state=stream.state_dict();state['total']+=1
        with self.assertRaises(ValueError):stream.load_state_dict(state)
    def test_inconsistent_distributed_state_rejected(self):
        with self.assertRaises(ValueError):canonical_cursor([{'count':0},{'count':8},{'count':9},{'count':10}],10)

if __name__=='__main__':unittest.main()
