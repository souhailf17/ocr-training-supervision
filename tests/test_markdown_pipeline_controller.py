import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from markdown_pipeline_controller import validate_manifest,NAMES

class Scope(unittest.TestCase):
    def document(self):
        m={'namespace':'mtrna-ns1','labels':{'project':'q35ocr-20260921-p560'}}
        return {'items':[{'kind':'ConfigMap','metadata':{**m,'name':NAMES['training']+'-code'}},
                         {'kind':'Job','metadata':{**m,'name':NAMES['training']},'spec':{'backoffLimit':0,'template':{'spec':{}}}}]}
    def test_own_scoped_job_accepted(self):validate_manifest(self.document(),'training')
    def test_unrelated_resource_rejected(self):
        d=self.document();d['items'][1]['metadata']['name']='colleague-job'
        with self.assertRaises(ValueError):validate_manifest(d,'training')
    def test_node_pinning_rejected(self):
        d=self.document();d['items'][1]['spec']['template']['spec']['nodeName']='worker7'
        with self.assertRaises(ValueError):validate_manifest(d,'training')
    def test_automatic_restart_rejected(self):
        d=self.document();d['items'][1]['spec']['backoffLimit']=3
        with self.assertRaises(ValueError):validate_manifest(d,'training')

if __name__=='__main__':unittest.main()
