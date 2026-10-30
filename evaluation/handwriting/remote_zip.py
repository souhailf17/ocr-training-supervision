"""Bounded HTTP-range reads of public ZIPs; never fetch an entire large archive."""
import io,urllib.request,zipfile

class RemoteZipFile(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.downloaded=0;self.cache_start=-1;self.cache=b''
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative ZIP offset')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if n>24000000:raise ValueError('Refusing an unexpectedly large archive member')
        start=self.pos
        if self.cache_start<=start and start+n<=self.cache_start+len(self.cache):
            offset=start-self.cache_start;self.pos+=n
            return self.cache[offset:offset+n]
        requested=min(max(n,65536),self.size-start);end=start+requested-1
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','User-Agent':'Public-OCR-Evaluation/1.0'})
        with urllib.request.urlopen(req,timeout=65) as r:
            if r.status!=206:raise RuntimeError('Public archive does not support bounded HTTP ranges')
            content_range=r.headers.get('Content-Range','')
            if not content_range.startswith(f'bytes {start}-'):raise RuntimeError('Unexpected cached byte range')
            data=r.read(requested+1)
        if len(data)!=requested:raise RuntimeError('Incomplete bounded archive read')
        self.pos+=n;self.downloaded+=len(data);self.cache_start=start;self.cache=data
        return data[:n]


if __name__=='__main__':
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parent/'sources'
    for name,record,key in [('rimes-zip','rimes-record.json','Images_Courriers.zip'),('rev-city-zip','rev-city-record.json','revcity-htr.zip')]:
        d=json.loads((root/record).read_text());f=next(x for x in d['files'] if x['key']==key)
        try:
            stream=RemoteZipFile(f['links']['self'],f['size'])
            with zipfile.ZipFile(stream) as z:
                files=[{'name':i.filename,'bytes':i.file_size,'compressed':i.compress_size} for i in z.infolist()]
            (root/(name+'-index.json')).write_text(json.dumps(files,indent=2),encoding='utf-8')
            print(name,len(files),'index_download_bytes',stream.downloaded,'examples',files[:15])
        except Exception as e:print(name,type(e).__name__,str(e)[:120])
