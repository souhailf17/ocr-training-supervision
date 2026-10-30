"""Literal OCR instructions and auditable failure flags, never text repair.

These checks do not establish image grounding or guarantee hallucination-free
OCR. An interrupted/flagged response must not be presented as complete OCR.
"""
import re
import unicodedata

TRAINING_MARKDOWN_PROMPT = (
    'Transcribe all visible text in this image into Markdown, preserving its language, '
    'mathematical notation and reading order. Include every table at its position as '
    'a complete Markdown pipe table with every header, row and cell. For merged cells, '
    'write the text once in its anchor cell and leave covered cells empty. '
    'If there is no header row, use an empty Markdown header before all data rows. '
    'Do not replace tables with links or placeholders. Do not translate, summarize, '
    'explain or invent content. If the page is blank, output nothing.'
)

LITERAL_RULES = (
    'Copy only marks and text actually visible on this single page. '
    'Stop immediately after its last visible item. Never append another document, '
    'a translation, a continuation, a greeting, an explanation or a guessed word. '
    'Keep the original writing system and language of each item. '
    'Copy digits exactly as drawn: 0123456789, ٠١٢٣٤٥٦٧٨٩ and ۰۱۲۳۴۵۶۷۸۹ '
    'are different glyph families; never convert between them. '
    'Keep accents, diacritics, decimal separators, punctuation, currencies and '
    'mathematical symbols exactly as visible. Do not correct spelling or numbers. '
    'Represent an unreadable span by [?], without guessing. '
    'If the page is blank, output nothing.'
)

def literal_prompt(original, language='mixed'):
    if language not in {'ar','fr','en','mixed'}:
        raise ValueError('Language must describe the inspected page')
    hint = {
        'fr': 'The page has been identified as French. Do not add Arabic text or Arabic punctuation to French text.',
        'en': 'The page has been identified as English. Do not add Arabic text or Arabic punctuation to English text.',
        'ar': 'The page contains Arabic. Preserve any Latin text actually visible as well.',
        'mixed': 'The page may be multilingual; retain each visible language without translating.',
    }[language]
    return original + '\n\n' + LITERAL_RULES + '\n' + hint

def is_exact_white_blank(image):
    """Only a uniform near-white RGB image; never threshold away faint text."""
    if image.mode!='RGB':raise ValueError('Inspect the actual RGB input image')
    return all(low==high and low>=250 for low,high in image.getextrema())

def arabic_letters(text):
    return sum(unicodedata.category(c).startswith('L') and 'ARABIC' in unicodedata.name(c, '') for c in text)

class StreamingGuard:
    """Remember why a response was stopped; do not discard its raw text."""
    def __init__(self, language='mixed'):
        if language not in {'ar','fr','en','mixed'}:raise ValueError('Invalid page language')
        self.language=language;self.reason=None
    def check(self,text):
        if self.reason is not None:return True
        if repetition_flags(text):self.reason='repetition'
        elif self.language in {'fr','en'} and arabic_letters(text)>=4:self.reason='unexpected_arabic_text'
        return self.reason is not None

def repetition_flags(text):
    """Conservative failure flags; repeated legitimate forms can also be flagged."""
    found=[]
    if re.search(r'([^\w\s])\1{63,}',text):found.append('punctuation_run')
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    # Includes multi-line cycles, not only identical consecutive lines.
    for period in range(1,min(12,len(lines)//3)+1):
        block=lines[-period:]
        if len(''.join(block))>=24 and block==lines[-2*period:-period]==lines[-3*period:-2*period]:
            found.append('repeated_line_cycle');break
    # A long loop without newlines, while avoiding normal short repeated words.
    tail=text[-4096:]
    for period in range(32,min(512,len(tail)//4)+1):
        block=tail[-period:]
        if len(set(block))>=4 and tail.endswith(block*4):
            found.append('repeated_text_cycle');break
    return found

def inspect_output(text, language='mixed', stop_reason='eos'):
    flags=repetition_flags(text)
    if language in {'fr','en'}:
        if arabic_letters(text)>=4:flags.append('unexpected_arabic_text')
        if sum(text.count(c) for c in '،؛؟')>=3:flags.append('unexpected_arabic_punctuation')
    if '[?]' in text:flags.append('unreadable_span')
    if stop_reason not in {'eos','stop'}:flags.append('incomplete_generation')
    return {
        'requires_review':bool(flags),'flags':sorted(set(flags)),
        'text_rewritten':False,'image_grounding_verified':False,
        'raw_characters':len(text),
        'digit_glyph_counts':{
            name:sum(c in alphabet for c in text) for name,alphabet in (
                ('latin','0123456789'),('arabic_indic','٠١٢٣٤٥٦٧٨٩'),('eastern_arabic_indic','۰۱۲۳۴۵۶۷۸۹'))},
    }

def digit_substitutions(reference, prediction):
    """Reference-only diagnostic. No conversion of the returned OCR text."""
    pattern=r'[0-9٠-٩۰-۹]+'
    ref=re.findall(pattern,reference);pred=re.findall(pattern,prediction)
    def value(s):return ''.join(str(unicodedata.decimal(c)) for c in s)
    # Explicitly keep positional alignment as a diagnostic, not a correctness proof.
    changes=[]
    for i,(a,b) in enumerate(zip(ref,pred)):
        if value(a)==value(b) and a!=b:changes.append({'position':i,'reference':a,'prediction':b})
    return {'reference_numbers':len(ref),'predicted_numbers':len(pred),'aligned_same_value_different_glyphs':changes,'alignment':'positional; insertions/deletions can shift matches'}
