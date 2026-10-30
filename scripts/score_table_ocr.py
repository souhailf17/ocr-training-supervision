"""Score real saved OCR against complete annotations, including table cells.

Reference tables come ONLY from the reference annotation. Prediction tables come
ONLY from the prediction. A predicted file link is never filled with ground truth.
Cell scores use strict (table,row,cell) order; they are not TEDS.
"""
import argparse
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from markdown_it import MarkdownIt
from rapidfuzz.distance import Levenshtein
from annotation_targets import resolve_annotation, parse_tables, table_references, TableTargetError

PARSER=MarkdownIt('commonmark',{'html':True}).enable(['table','strikethrough'])


class Visible(HTMLParser):
    BLOCKS={'p','div','br','hr','li','ul','ol','table','tr','td','th','h1','h2','h3','h4','h5','h6','blockquote','pre','section'}
    def __init__(self):
        super().__init__(convert_charrefs=True); self.parts=[]; self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in {'script','style'}:self.hidden+=1
        if tag in self.BLOCKS and not self.hidden:self.parts.append(' ')
    def handle_endtag(self,tag):
        if tag in {'script','style'}:self.hidden=max(0,self.hidden-1)
        if tag in self.BLOCKS and not self.hidden:self.parts.append(' ')
    def handle_data(self,text):
        if not self.hidden:self.parts.append(text)


def normalize(text):
    text=unicodedata.normalize('NFKC',text)
    text=''.join(c for c in text if c!='\u0640' and unicodedata.category(c)!='Cf')
    return ' '.join(text.split())


def unwrap(text):
    match=re.fullmatch(r'\s*```(?:html|markdown|md|json)?\s*\n(.*?)\n```\s*',text,re.S|re.I)
    return match.group(1) if match else text


def plain(text):
    visible=Visible();visible.feed(PARSER.render(unwrap(text)));visible.close()
    return normalize(''.join(visible.parts))


def edit(reference,prediction):
    distance=Levenshtein.distance(reference,prediction)
    return {'reference_units':len(reference),'prediction_units':len(prediction),
            'errors':distance,'rate':distance/len(reference) if reference else None}


def structures(text):
    # CommonMark renders pipe tables as HTML; inline HTML stays inline.
    return parse_tables(PARSER.render(unwrap(text)))


def shape(table):
    return [[(c['tag'],c['rowspan'],c['colspan']) for c in row] for row in table['rows']]


def cells_by_slot(tables):
    return {(t,r,c):normalize(cell['text']) for t,table in enumerate(tables)
            for r,row in enumerate(table['rows']) for c,cell in enumerate(row)}


def score(annotation,prediction):
    reference=resolve_annotation(annotation)
    predicted_annotation_error=None
    if isinstance(prediction,dict):
        try:prediction=resolve_annotation(prediction).text
        except TableTargetError as exc:
            predicted_annotation_error=str(exc)
            prediction=prediction.get('markdown','')
    if not isinstance(prediction,str):raise ValueError('Prediction must be saved Markdown/HTML text or its own annotation JSON')
    ref_text,pred_text=plain(reference.text),plain(prediction)
    ref_tables=structures(reference.text)
    prediction_table_error=None
    try:pred_tables=structures(prediction)
    except TableTargetError as exc:
        pred_tables=[];prediction_table_error=str(exc)
    ref_cells,pred_cells=cells_by_slot(ref_tables),cells_by_slot(pred_tables)
    table_errors=reference_units=prediction_units=0
    word_errors=reference_words=prediction_words=0
    for slot in set(ref_cells)|set(pred_cells):
        a,b=ref_cells.get(slot,''),pred_cells.get(slot,'')
        table_errors+=Levenshtein.distance(a,b);reference_units+=len(a);prediction_units+=len(b)
        aw,bw=a.split(),b.split()
        word_errors+=Levenshtein.distance(aw,bw);reference_words+=len(aw);prediction_words+=len(bw)
    malformed=bool(prediction_table_error or predicted_annotation_error)
    return {'whole_page':{'CER':edit(ref_text,pred_text),'WER':edit(ref_text.split(),pred_text.split())},
            'tables':{'reference_tables':len(ref_tables),'predicted_tables':None if malformed else len(pred_tables),
                      'reference_cells':len(ref_cells),'predicted_cells':None if malformed else len(pred_cells),
                      'exact_structure_matches':None if malformed else sum(shape(a)==shape(b) for a,b in zip(ref_tables,pred_tables)),
                      'exact_cell_text_matches':None if malformed else sum(ref_cells[k]==pred_cells.get(k) for k in ref_cells),
                      'cell_CER':None if malformed else {'reference_units':reference_units,'prediction_units':prediction_units,'errors':table_errors,'rate':table_errors/reference_units if reference_units else None},
                      'cell_WER':None if malformed else {'reference_units':reference_words,'prediction_units':prediction_words,'errors':word_errors,'rate':word_errors/reference_words if reference_words else None},
                      'unresolved_prediction_links':len(list(table_references(prediction))),
                      'prediction_table_parse_error':prediction_table_error,
                      'prediction_annotation_error':predicted_annotation_error},
            'reference_audit':reference.audit,
            'method':{'text_normalization':'CommonMark/HTML visible text, NFKC, whitespace, tatweel/invisible controls; preserve diacritics, letters, numbers and repetitions.',
                      'cell_alignment':'Strict ordered (table,row,cell) slots; missing cells are deletions, extra cells insertions. No TEDS claim.',
                      'no_reference_injection':'Never resolve prediction links with reference tables. A prediction JSON may use only its own tables.',
                      'malformed_tables':'Do not publish a table rate/count when prediction HTML or prediction references are invalid.'}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotation',type=Path,required=True)
    parser.add_argument('--prediction',type=Path,required=True)
    parser.add_argument('--prediction-json',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    annotation=json.loads(args.annotation.read_text(encoding='utf-8'))
    raw=args.prediction.read_text(encoding='utf-8')
    prediction=json.loads(raw) if args.prediction_json else raw
    result=score(annotation,prediction)
    result['sources']={'annotation':str(args.annotation),'prediction':str(args.prediction),'new_inference':False}
    if args.output.exists():raise ValueError('Refusing to overwrite an existing score file')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
