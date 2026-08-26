"""Answer metrics used by the MuSiQue experiment runner."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = Counter(predicted) & Counter(expected)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def exact_match_any(prediction: str, answers: Iterable[str]) -> bool:
    return any(exact_match(prediction, answer) for answer in answers)


def token_f1_any(prediction: str, answers: Iterable[str]) -> float:
    scores = [token_f1(prediction, answer) for answer in answers]
    return max(scores, default=0.0)
