"""Token estimation (M2 groundwork).

Ollama exposes no tokenize endpoint, so we estimate before sending and
record the real counts (prompt_eval_count / eval_count) afterwards.

A plain chars/4 rule badly underestimates languages written in CJK scripts
(roughly one token per character), which matters for a language tutor, so
those characters are counted separately.
"""

import re

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
CHARS_PER_TOKEN = 3.5  # conservative for latin scripts
MESSAGE_OVERHEAD = 4  # chat template tokens around each message


def estimate_tokens(text: str) -> int:
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return cjk + int(other / CHARS_PER_TOKEN) + 1


def estimate_message(content: str) -> int:
    return estimate_tokens(content) + MESSAGE_OVERHEAD
