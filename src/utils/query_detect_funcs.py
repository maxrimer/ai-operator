import re
from typing import List
import pymorphy3


QUESTION_MARK = re.compile(r"\?\s*$")
morph = pymorphy3.MorphAnalyzer()


def looks_like_command(text: str, kw_nouns: List) -> bool:
    kw_lemmas = {morph.parse(w)[0].normal_form for w in kw_nouns}
    re_words = re.compile(r"[а-яё]+", re.I)
    words = re_words.findall(text.lower())
    if not words:
        return False

    first = morph.parse(words[0])[0]

    if 'VERB' in first.tag and 'impr' in first.tag:
        for w in words[1:]:
            lemma = morph.parse(w)[0].normal_form
            if lemma in kw_lemmas:
                return True
    return False


def is_potential_query(text: str, kw_nouns: List) -> bool:
    text = text.strip()
    if QUESTION_MARK.search(text):
        return True
    if looks_like_command(text, kw_nouns):
        return True
    if len(text) >= 8 and any(w in text.lower() for w in kw_nouns):
        return True
    return False




