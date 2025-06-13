import re
# import torch
from typing import List
# from models.toxicity import tok, mdl
from models.lang_detect import detect_lang

RUDE_WORDS: List[str] = ["лох","тупой","идиот","черт"]
MAX_LEN_CHARS, MIN_LEN_WORDS = 500, 3
_re_multi = re.compile(r"[!?]{3,}")


def hint_validator_node(state):
    hint = (state.hint or "").strip()
    problems = []

    if len(hint) > MAX_LEN_CHARS:
        problems.append("too_long")
    if len(hint.split()) < MIN_LEN_WORDS:
        problems.append("too_short")
    if _re_multi.search(hint):
        problems.append("spam_punct")

    if detect_lang(hint) != "ru":
        problems.append("non_russian")

    if any(bad in hint.lower() for bad in RUDE_WORDS):
        problems.append("rude_word")

    # with torch.no_grad():
    #     toks = tok(hint, return_tensors="pt", truncation=True, max_length=128)
    #     tox_score = torch.softmax(mdl(**toks).logits, dim=1)[0][1].item()
    # if tox_score > 0.5:
    #     problems.append("impolite_tone")

    state.hint_valid = not problems
    state.validator_msg = ",".join(problems) if problems else None
    return state

