import re,html,unicodedata
from html.parser import HTMLParser
import metrics as measures

class VisibleHTML(HTMLParser):
 def __init__(self):super().__init__(convert_charrefs=True);self.parts=[];self.skip=0
 def handle_starttag(self,tag,attrs):
  if tag in ('script','style'):self.skip+=1
  if tag in ('td','th','tr','p','div','br','li','h1','h2','h3','h4'):self.parts.append(' ')
 def handle_endtag(self,tag):
  if tag in ('script','style'):self.skip=max(0,self.skip-1)
  if tag in ('td','th','tr','p','div','li','h1','h2','h3','h4'):self.parts.append(' ')
 def handle_data(self,data):
  if not self.skip:self.parts.append(data)

def visible(text):
 text=re.sub(r'<!--.*?-->','',text,flags=re.S)
 if re.search(r'</?(?:table|tr|td|th|div|span|p|br|html|body)\b',text,re.I):
  parser=VisibleHTML();parser.feed(text);text=''.join(parser.parts)
 lines=[]
 for line in text.splitlines():
  if re.match(r'^\s*(```|~~~)',line):continue
  if re.fullmatch(r'\s*\|?[\s:|-]+\|?\s*',line) and '-' in line:continue
  line=re.sub(r'^\s{0,3}#{1,6}\s+','',line);line=re.sub(r'^\s*>\s?','',line)
  line=re.sub(r'^\s*[-+*]\s+','',line);line=re.sub(r'!\[[^\]]*\]\([^)]*\)','',line)
  line=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',line);line=re.sub(r'(\*\*|__)(.*?)\1',r'\2',line)
  line=re.sub(r'`([^`]+)`',r'\1',line)
  if '|' in line:line=' '.join(c.strip() for c in line.strip().strip('|').split('|'))
  lines.append(line)
 return ' '.join(unicodedata.normalize('NFC',html.unescape('\n'.join(lines))).split())

def score(ref,text):return measures.score_text(visible(ref),visible(text))
