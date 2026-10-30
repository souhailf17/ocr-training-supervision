import base64,os,sys,zipfile,io
from pathlib import Path
root=Path('/tmp/markdown-deps');root.mkdir(exist_ok=True)
blob=base64.b64decode(Path('/corpus-code/markdown-deps.b64').read_text())
with zipfile.ZipFile(io.BytesIO(blob)) as z:
 for item in z.infolist():
  target=(root/item.filename).resolve()
  assert target.is_relative_to(root.resolve())
  if not item.is_dir():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
sys.path.insert(0,str(root));sys.path.insert(0,'/corpus-code')
import prepare_markdown_corpus
prepare_markdown_corpus.main()
