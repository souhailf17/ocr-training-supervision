"""Decode nonprinted Unlimited-OCR-2 BLOCK/FORK headers, keeping all content.

Grammar/coordinate scale follows the official pinned studio.html parseBlocks.
No spelling, Unicode, digit, language, header/footer or unreadable-span repair.
Unparsed material is retained and flagged rather than silently suppressed.
Single-page evaluation keeps native block order; no content-based reordering.
"""
import base64,html,re
from pathlib import Path
PAGE=re.compile(r'<\s*/?\s*PAGE\s*\d*\s*>',re.I)
BLOCK=re.compile(r'<\s*BLOCK\s*>',re.I)
MERGE=re.compile(r'<\s*/?\s*MERGE\s*>',re.I)
HEADER=re.compile(r'^\s*(?P<label>[A-Za-z_]+)\s*(?:(?P<level1><\s*/?\s*LEVEL\s*\d*\s*[>}]?)\s*)?(?:\[\s*(?P<coords>-?\d+(?:\s*[,，]\s*-?\d+){3,})\s*\]\s*)?(?:(?P<level2><\s*/?\s*LEVEL\s*\d*\s*[>}]?)\s*)?<\s*FORK\s*>',re.I)
def decode(raw):
    blocks=[];flags=[];parts=[]
    segments=PAGE.split(raw)
    if segments and not segments[0].strip():segments=segments[1:]
    for page,segment in enumerate(segments):
        pieces=BLOCK.split(segment)
        prefix=MERGE.sub('',pieces[0]).strip()
        if prefix:
            parts.append(prefix);blocks.append({'type':'unparsed','box':None,'page':page,'content':prefix});flags.append('unparsed_prefix_or_output')
        for piece in pieces[1:]:
            match=HEADER.match(piece)
            if not match:
                parts.append('<BLOCK>'+piece.strip());blocks.append({'type':'unparsed','box':None,'page':page,'content':piece.strip()});flags.append('unparsed_block_header');continue
            content=MERGE.sub('',piece[match.end():]).strip()
            numbers=[int(x.strip()) for x in re.split('[,，]',match['coords'])] if match['coords'] else []
            box=numbers[:4] if numbers else None
            if box and (not 0<=box[0]<=box[2]<=1000 or not 0<=box[1]<=box[3]<=1000):flags.append('invalid_box_coordinates')
            if not box:flags.append('block_without_coordinates')
            blocks.append({'type':match['label'].lower(),'box':box,'page':page,'content':content})
            if content:parts.append(content)
    return {'text':'\n\n'.join(parts),'blocks':blocks,'flags':sorted(set(flags)),
      'transformation':'remove nonprinted PAGE/BLOCK/FORK/LEVEL/MERGE structure only; retain all block contents and native order',
      'headers_footers_page_numbers_retained':True,'glyphs_or_words_rewritten':False}
def overlay(image_path,decoded,output):
    from PIL import Image
    with Image.open(image_path) as image:w,h=image.size
    colors={'text':'#1769aa','table':'#14843b','title':'#c6253e','header':'#6b7280','footer':'#6b7280','page_number':'#6b7280','image':'#bf8700'}
    href='data:image/png;base64,'+base64.b64encode(Path(image_path).read_bytes()).decode()
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><title>Blocs prédits par Unlimited-OCR-2 ; fidélité non garantie</title><image href="{href}" width="{w}" height="{h}"/>']
    count=0
    for i,b in enumerate(decoded['blocks'],1):
        if not b['box'] or b['page']!=0:continue
        x0,y0,x1,y1=b['box']
        if not 0<=x0<=x1<=1000 or not 0<=y0<=y1<=1000:continue
        color=colors.get(b['type'],'#8844aa');x=x0*w/1000;y=y0*h/1000;bw=(x1-x0)*w/1000;bh=(y1-y0)*h/1000
        parts.append(f'<g><title>{i}. {html.escape(b["type"])}: {html.escape(b["content"][:160])}</title><rect x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{bh:.2f}" fill="{color}" fill-opacity=".08" stroke="{color}" stroke-width="1.5"/><text x="{x:.2f}" y="{max(10,y-2):.2f}" fill="{color}" font-size="10" font-family="Arial">{i}</text></g>');count+=1
    parts.append('</svg>');Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(''.join(parts),encoding='utf-8');return count
