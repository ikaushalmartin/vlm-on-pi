"""Work out which item a plain-English question is about.

"where are my car keys?" -> "car keys". No language model is needed: the
answer is always one of the items the user registered, so the question is
matched against those names. The longest match wins, so "car keys" beats
"keys" when both are registered.
"""
from .detector import normalize


def item_in_question(question, items):
    text = " %s " % normalize(question)
    best = None
    for item in items:
        name = normalize(item)
        if name and (" %s " % name) in text:
            if best is None or len(name) > len(normalize(best)):
                best = item
    return best
