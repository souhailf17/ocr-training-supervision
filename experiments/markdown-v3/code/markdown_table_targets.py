"""Versioned HTML -> GFM targets. No source rewriting, external fetch or inference.

Merged cell text appears once at its DOM anchor. Covered slots remain empty;
the loss of span geometry inherent in Markdown is recorded in the audit.
"""
import hashlib
import re
from collections import defaultdict, deque
from annotation_targets import (TableHTML, TableTargetError, ResolvedTarget,
                                resolve_annotation, assert_no_table_placeholders)

TARGET_FORMAT = 'markdown-pipe-tables-v3'
VERSION = 'ocr-markdown-v3-orig-aug1'
PROMPT = (
    'Transcribe all visible text in this image into Markdown, preserving its language, '
    'mathematical notation and reading order. Include every table at its position as '
    'a complete Markdown pipe table with every header, row and cell. For merged cells, '
    'write the text once in its anchor cell and leave covered cells empty. '
    'If there is no header row, use an empty Markdown header before all data rows. '
    'Do not replace tables with links or placeholders. Do not translate, summarize, '
    'explain or invent content. If the page is blank, output nothing.'
)

def sha(raw):
    return hashlib.sha256(raw.encode('utf-8') if isinstance(raw,str) else raw).hexdigest()

def visible(text):
    return ' '.join(text.split())

class StructuredHTML(TableHTML):
    def __init__(self):
        super().__init__(); self.group=0; self.active_group=0
    def handle_starttag(self, tag, attrs):
        if tag in {'thead','tbody','tfoot'}:
            self.group+=1; self.active_group=self.group
        if self.cell is not None and tag in {'p','div','li','hr'}:
            self.cell['text']+=' '
        super().handle_starttag(tag,attrs)
        if tag=='table': self.table['row_groups']=[]
        if tag=='tr': self.table['row_groups'].append(self.active_group)
    def handle_endtag(self, tag):
        if self.cell is not None and tag in {'p','div','li'}:
            self.cell['text']+=' '
        super().handle_endtag(tag)
        if tag in {'thead','tbody','tfoot'}:
            self.group+=1; self.active_group=self.group

def escape(text):
    # Literal cell text: prevent accidental emphasis, links, HTML or columns.
    text=visible(text).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    return re.sub(r'([\\`*_\[\]|])',r'\\\1',text)

def convert_table(content):
    parser=StructuredHTML();parser.feed(content);parser.close()
    if parser.stack or len(parser.tables)!=1:
        raise TableTargetError('expected_one_balanced_table')
    table=parser.tables[0]; rows=table['rows']; groups=table['row_groups']
    if not rows or not any(rows): raise TableTargetError('table_has_no_cells')
    if len(rows)>1000:raise TableTargetError('table_row_budget_exceeded')
    occupied={};anchors=[];maximum=0
    for r,row in enumerate(rows):
        c=0
        group_end=r+1
        while group_end<len(rows) and groups[group_end]==groups[r]:group_end+=1
        for cell in row:
            while (r,c) in occupied:c+=1
            rs=group_end-r if cell['rowspan']==0 else cell['rowspan']; cs=cell['colspan']
            if r+rs>group_end:raise TableTargetError('rowspan_crosses_row_group')
            if c+cs>256:raise TableTargetError('table_column_budget_exceeded')
            for rr in range(r,r+rs):
                for cc in range(c,c+cs):
                    if (rr,cc) in occupied:raise TableTargetError('overlapping_table_spans')
                    occupied[rr,cc]=(r,c)
            anchors.append({'row':r,'column':c,'rowspan':rs,'source_rowspan':cell['rowspan'],
                            'colspan':cs,'tag':cell['tag'],'text':visible(cell['text'])})
            maximum=max(maximum,c+cs);c+=cs
    grid=[['']*maximum for _ in rows]
    for anchor in anchors:grid[anchor['row']][anchor['column']]=anchor['text']
    # All-th, non-rowspanning first row has true header semantics. Otherwise
    # prepend an empty header instead of relabelling a source data row.
    true_header=bool(rows[0]) and all(c['tag']=='th' and c['rowspan']==1 for c in rows[0])
    rendered=grid if true_header else [['']*maximum]+grid
    def line(cells):return '| '+' | '.join(escape(c) for c in cells)+' |'
    output=[line(rendered[0]),'| '+' | '.join(['---']*maximum)+' |']
    output.extend(line(r) for r in rendered[1:])
    caption=visible(table['caption'])
    text=(escape(caption)+'\n\n' if caption else '')+'\n'.join(output)
    audit={'rows':len(rows),'columns':maximum,'source_cells':len(anchors),'anchors':anchors,
           'source_html_sha256':sha(content),'markdown_sha256':sha(text),
           'synthetic_empty_header':not true_header,'caption':caption,
           'merged_cells':sum(a['rowspan']!=1 or a['colspan']!=1 for a in anchors),
           'span_policy':'anchor-once-covered-slots-empty',
           'column_policy':'preserve-source-DOM-order; RTL display uses direction:rtl'}
    return text,audit

def resolve_markdown(annotation):
    validated=resolve_annotation(annotation)
    entries={t['id']:t['content'] for t in annotation.get('tables',[])}
    ids=defaultdict(deque)
    for ident in validated.audit['resolved_table_ids']:ids[entries[ident]].append(ident)
    audit=[]
    def substitute(match):
        content=match.group()
        if not ids[content]:raise TableTargetError('inline_table_missing_from_json_tables')
        ident=ids[content].popleft();md,detail=convert_table(content)
        audit.append({'id':ident,**detail})
        return '\n\n'+md+'\n\n'
    text=re.sub(r'<table\b[^>]*>.*?</table\s*>',substitute,validated.text,flags=re.S|re.I)
    if any(ids.values()):raise TableTargetError('table_not_converted')
    assert_no_table_placeholders(text)
    if re.search(r'<(?:table|tr|td|th)\b',text,re.I):raise TableTargetError('html_table_remains')
    if sum(t['source_cells'] for t in audit)!=validated.audit['cell_count']:
        raise TableTargetError('cell_count_changed')
    result={**validated.audit,'target_format':TARGET_FORMAT,'tables':audit,
            'markdown_sha256':sha(text),'merged_cell_geometry_preserved_in':'audit only; pipe tables cannot express spans',
            'normalization':'cell whitespace collapsed; every literal cell text retained once'}
    return ResolvedTarget(text,result)
