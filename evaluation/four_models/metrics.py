"""Exact Unicode Levenshtein and CER/WER functions used in this experiment."""
import unicodedata

def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())

def distance(left, right):
    """Exact unit-cost Levenshtein, bit-parallel; works with chars or word lists."""
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

def score_text(reference, text):
    reference, text = normalize(reference), normalize(text)
    if not reference:
        return None
    words = reference.split()
    char_errors, word_errors = distance(reference, text), distance(words, text.split())
    return {'cer': 100 * char_errors / len(reference), 'wer': 100 * word_errors / len(words),
            'character_errors': char_errors, 'word_errors': word_errors,
            'reference_characters': len(reference), 'reference_words': len(words)}
