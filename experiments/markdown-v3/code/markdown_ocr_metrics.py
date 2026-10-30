"""Independent normalized OCR metrics against complete Markdown supervision.

Predicted links are NEVER resolved with reference content. Strict cell slots are
reported with that limitation; these scores are not TEDS or table detection AP.
"""
import re,unicodedata
from html.parser import HTMLParser
from markdown_it import MarkdownIt
from annotation_targets import parse_tables,table_references,TableTargetError

PARSER=MarkdownIt('commonmark',{'html':True}).enable(['table','strikethrough'])
class Visible(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.parts=[];self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in {'script','style'}:self.hidden+=1
        if tag in {'p','div','br','hr','li','table','tr','td','th','h1','h2','h3','blockquote'} and not self.hidden:self.parts.append(' ')
    def handle_endtag(self,tag):
        if tag in {'script','style'}:self.hidden=max(0,self.hidden-1)
        if tag in {'p','div','li','table','tr','td','th','h1','h2','h3','blockquote'} and not self.hidden:self.parts.append(' ')
    def handle_data(self,data):
        if not self.hidden:self.parts.append(data)
def normalize(text):
    text=unicodedata.normalize('NFKC',text)
    return ' '.join(''.join(c for c in text if c!='\u0640' and unicodedata.category(c)!='Cf').split())
def unwrap(text):
    m=re.fullmatch(r'\s*```(?:markdown|md|html)?\s*\n(.*?)\n```\s*',text,re.S|re.I)
    return m.group(1) if m else text
def plain(text):
    v=Visible();v.feed(PARSER.render(unwrap(text)));v.close();return normalize(''.join(v.parts))
def distance(a,b):
    if len(a)<len(b):a,b=b,a
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]
def slots(tables):
    return {(t,r,c):normalize(cell['text']) for t,table in enumerate(tables) for r,row in enumerate(table['rows']) for c,cell in enumerate(row)}
def score(reference,prediction):
    ref=plain(reference);pred=plain(prediction);rt=parse_tables(PARSER.render(reference));error=None
    try:pt=parse_tables(PARSER.render(unwrap(prediction)))
    except TableTargetError as exc:pt=[];error=str(exc)
    rc,pc=slots(rt),slots(pt);characters=sum(len(v) for v in rc.values())
    ce=distance(ref,pred);we=distance(ref.split(),pred.split())
    result={'character_errors':ce,'reference_characters':len(ref),'CER':ce/len(ref) if ref else None,
            'word_errors':we,'reference_words':len(ref.split()),'WER':we/len(ref.split()) if ref.split() else None,
            'reference_tables':len(rt),'predicted_tables':len(pt),'unresolved_prediction_table_links':len(list(table_references(prediction))),
            'reference_nonempty_cells':sum(bool(v) for v in rc.values()),'predicted_nonempty_cells':sum(bool(v) for v in pc.values()),
            'prediction_table_parse_error':error,'alignment':'strict canonical Markdown (table,row,column) slots; not TEDS'}
    if error:result.update(strict_cell_CER=None,missing_nonempty_cells=None)
    else:
        errors=sum(distance(rc.get(k,''),pc.get(k,'')) for k in rc.keys()|pc.keys())
        result.update(strict_cell_character_errors=errors,reference_cell_characters=characters,
                      strict_cell_CER=errors/characters if characters else None,
                      missing_nonempty_cells=sum(bool(v) and not pc.get(k,'') for k,v in rc.items()))
    return result
