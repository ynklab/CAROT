"""
Answer normalization used for generative XQuAD evaluation.

Adapted from the SQuAD v1.1 and MLQA answer-normalization procedures,
with language-specific modifications for the languages evaluated here.
"""
 
import re
import string
import unicodedata
 
# All Unicode punctuation (not just ASCII) -- needed for Greek/Cyrillic scripts.
PUNCT = {chr(i) for i in range(0x110000) if unicodedata.category(chr(i)).startswith('P')}
PUNCT |= set(string.punctuation)
 
# Only strip articles where a simple regex is safe and standard practice
# (MLQA precedent: en, es, vi). Greek is deliberately excluded; Russian has
# no articles.
ARTICLES_REGEX = {
    'English': re.compile(r'\b(a|an|the)\b', re.IGNORECASE),
    'Spanish (Latin American)': re.compile(r'\b(el|la|los|las|un|una|unos|unas)\b', re.IGNORECASE),
    'Vietnamese': re.compile(r'\b(một)\b', re.IGNORECASE),
}
 
 
def remove_punctuation(text: str) -> str:
    return ''.join(ch for ch in text if ch not in PUNCT)
 
 
def remove_articles(text: str, lang: str) -> str:
    pattern = ARTICLES_REGEX.get(lang)
    if pattern is None:
        return text
    return pattern.sub(' ', text)
 
 
def white_space_fix(text: str) -> str:
    return ' '.join(text.split())
 
 
def normalize_answer(s: str, lang: str) -> str:
    """Lowercase, strip punctuation, strip articles (where applicable),
    collapse whitespace. Language-aware (en, es, el, vi, zh, ru)."""
    s = s.lower()
    s = remove_punctuation(s)
    s = remove_articles(s, lang)
    s = white_space_fix(s)
    return s
