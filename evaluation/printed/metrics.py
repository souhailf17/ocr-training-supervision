"""Frozen historical scoring, independently portable and network-free.

NFC and collapsed whitespace; case, punctuation, diacritics and digit scripts
are retained. Identical visible-text extraction is applied to both systems.
"""
import html
import re
import unicodedata
from html.parser import HTMLParser


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if tag in ('td', 'th', 'tr', 'p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'h4'):
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)
        if tag in ('td', 'th', 'tr', 'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4'):
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def visible(text):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    if re.search(r'</?(?:table|tr|td|th|div|span|p|br|html|body)\b', text, re.I):
        parser = VisibleHTML()
        parser.feed(text)
        text = ''.join(parser.parts)
    lines = []
    for line in text.splitlines():
        if re.match(r'^\s*(```|~~~)', line):
            continue
        if re.fullmatch(r'\s*\|?[\s:|-]+\|?\s*', line) and '-' in line:
            continue
        line = re.sub(r'^\s{0,3}#{1,6}\s+', '', line)
        line = re.sub(r'^\s*>\s?', '', line)
        line = re.sub(r'^\s*[-+*]\s+', '', line)
        line = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', line)
        line = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', line)
        line = re.sub(r'(\*\*|__)(.*?)\1', r'\2', line)
        line = re.sub(r'`([^`]+)`', r'\1', line)
        if '|' in line:
            line = ' '.join(c.strip() for c in line.strip().strip('|').split('|'))
        lines.append(line)
    return ' '.join(unicodedata.normalize('NFC', html.unescape('\n'.join(lines))).split())


def distance(left, right):
    """Exact Levenshtein distance, unit costs, bit-parallel algorithm."""
    if len(left) > len(right):
        left, right = right, left
    n = len(left)
    if not n:
        return len(right)
    masks = {}
    for i, char in enumerate(left):
        masks[char] = masks.get(char, 0) | (1 << i)
    mask, top = (1 << n) - 1, 1 << (n - 1)
    positive, negative, score = mask, 0, n
    for char in right:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        score += bool(plus & top) - bool(minus & top)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = (minus | ~(vertical | plus)) & mask
        negative = (plus & vertical) & mask
    return score


def score(reference, prediction, *, visible_text=True):
    clean = visible if visible_text else lambda s: ' '.join(unicodedata.normalize('NFC', s).split())
    reference, prediction = clean(reference), clean(prediction)
    if not reference:
        raise ValueError('An empty reference cannot be scored as zero error')
    words = reference.split()
    ce = distance(reference, prediction)
    we = distance(words, prediction.split())
    return {'cer': 100 * ce / len(reference), 'wer': 100 * we / len(words),
            'character_errors': ce, 'word_errors': we,
            'reference_characters': len(reference), 'reference_words': len(words)}
