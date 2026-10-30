"""Deterministic full-corpus stream with disjoint DDP ownership and exact resume."""
import json,random
from markdown_table_targets import sha
from markdown_data_runtime import BUCKET,read,encode

class MarkdownStream:
    def __init__(self,reports,s3,processor,rank,world):
        self.reports=list(reports);random.Random(42).shuffle(self.reports)
        self.s3=s3;self.processor=processor;self.rank=rank;self.world=world
        self.total=sum(r['counts']['train'] for r in reports)
        self.shard=0;self.cursor=0;self.count=0;self.rows=None
    def state_dict(self):return {'shard':self.shard,'cursor':self.cursor,'count':self.count,'total':self.total,'seed':42}
    def load_state_dict(self,state):
        if state['total']!=self.total or state['seed']!=42 or not 0<=state['count']<=self.total:
            raise ValueError('Corpus/cursor changed')
        if not 0<=state['shard']<=len(self.reports) or state['cursor']<0:raise ValueError('Invalid cursor')
        self.shard=state['shard'];self.cursor=state['cursor'];self.count=state['count'];self.rows=None
    def __iter__(self):
        while self.shard<len(self.reports):
            if self.rows is None:
                m=self.reports[self.shard]['manifest'];raw=read(self.s3,m['key'])
                if sha(raw)!=m['sha256'] or len(raw)!=m['bytes']:raise ValueError('New training manifest changed')
                self.rows=[r for line in raw.splitlines() if (r:=json.loads(line))['split']=='train']
                if len(self.rows)!=self.reports[self.shard]['counts']['train']:raise ValueError('Shard train count differs')
                random.Random(42+self.shard).shuffle(self.rows)
                if self.cursor>len(self.rows):raise ValueError('Cursor outside shard')
            while self.cursor<len(self.rows):
                row=self.rows[self.cursor];owner=self.count%self.world
                if owner!=self.rank:
                    self.count+=1;self.cursor+=1;continue
                image=row['image'];start=image['offset']
                raw=read(self.s3,image['key']) if image.get('offset') is None else self.read_image(image,start)
                batch=encode(self.processor,row,raw,pin=True)
                if batch['input_ids'].shape[1]!=row['sequence_tokens'] or int((batch['labels'][:,1:]!=-100).sum())!=row['supervised_tokens']:
                    raise ValueError('Preparation/training encoder differs')
                self.count+=1;self.cursor+=1;yield batch
            self.shard+=1;self.cursor=0;self.rows=None
        if self.count!=self.total:raise ValueError('Full pass coverage incomplete')
    def read_image(self,image,start):
        from minio_retry import read_object
        return read_object(self.s3,BUCKET,image['key'],byte_range=f"bytes={start}-{start+image['size']-1}")

def canonical_cursor(states,global_samples):
    canonical=max(states,key=lambda s:s['count'])
    if canonical['count']!=global_samples or max(s['count'] for s in states)-min(s['count'] for s in states)>=len(states):
        raise ValueError('Distributed cursor/coverage differs')
    return dict(canonical)
