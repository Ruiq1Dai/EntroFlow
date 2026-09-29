"""Deterministic, auditable mathematical evaluator for MATH trajectories."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from tokenize import TokenError

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

EVALUATOR_VERSION = "math_equivalence_v2"
PARSER_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
MULTI_ANSWER_CUES = re.compile(
    r"\bfind\s+all\s+(?:possible\s+|real\s+)?(?:values|solutions|roots)\b|"
    r"\b(?:enter|list|give)\b.*\bseparated\s+by\s+commas?\b",
    flags=re.IGNORECASE | re.DOTALL,
)
UNIT_ALIASES = {
    "degree": "degree",
    "degrees": "degree",
    "dollar": "dollar",
    "dollars": "dollar",
    "cent": "cent",
    "cents": "cent",
    "percent": "percent",
    "percentage": "percent",
    "unit": "unit",
    "units": "unit",
    "squareunit": "unit^2",
    "squareunits": "unit^2",
    "cubicunit": "unit^3",
    "cubicunits": "unit^3",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "mm": "mm",
    "meter": "m",
    "meters": "m",
    "m": "m",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "inch": "inch",
    "inches": "inch",
    "foot": "foot",
    "feet": "foot",
    "yard": "yard",
    "yards": "yard",
    "mile": "mile",
    "miles": "mile",
    "second": "second",
    "seconds": "second",
    "minute": "minute",
    "minutes": "minute",
    "hour": "hour",
    "hours": "hour",
}


@dataclass(frozen=True)
class GoldSpec:
    canonical_answer: str
    components: tuple[str, ...]
    boxed_answers: tuple[str, ...]
    requires_multiple_answers: bool


@dataclass(frozen=True)
class EvaluationResult:
    correct: bool
    method: str
    component_methods: tuple[str, ...] = ()


def extract_all_boxed(text: str) -> list[str]:
    """Extract all balanced boxed/fbox payloads in source order."""
    matches: list[tuple[int, str]] = []
    for command in (r"\boxed", r"\fbox"):
        cursor = 0
        while True:
            start = text.find(command, cursor)
            if start < 0:
                break
            brace = text.find("{", start + len(command))
            if brace < 0:
                break
            depth = 0
            for index in range(brace, len(text)):
                escaped = index > 0 and text[index - 1] == "\\"
                if text[index] == "{" and not escaped:
                    depth += 1
                elif text[index] == "}" and not escaped:
                    depth -= 1
                    if depth == 0:
                        matches.append((start, text[brace + 1 : index].strip()))
                        cursor = index + 1
                        break
            else:
                cursor = len(text)
    return [value for _, value in sorted(matches)]


def last_boxed(text: str) -> str | None:
    answers = extract_all_boxed(text)
    return answers[-1] if answers else None


def extract_gold_answer(solution: str) -> str:
    """Legacy-compatible extraction of the final boxed answer."""
    answer = last_boxed(solution)
    if answer is None:
        raise ValueError("gold solution has no balanced \\boxed or \\fbox answer")
    return answer


def build_gold_spec(problem: str, solution: str) -> GoldSpec:
    boxed = extract_all_boxed(solution)
    if not boxed:
        raise ValueError("gold solution has no balanced \\boxed or \\fbox answer")
    requires_multiple = len(boxed) > 1 and bool(MULTI_ANSWER_CUES.search(problem))
    components = tuple(boxed if requires_multiple else boxed[-1:])
    return GoldSpec(
        canonical_answer=", ".join(components),
        components=components,
        boxed_answers=tuple(boxed),
        requires_multiple_answers=requires_multiple,
    )


def _strip_math_wrappers(value: str) -> str:
    value = value.strip().strip("`").strip()
    value = re.sub(r"^\*\*|\*\*$", "", value).strip()
    boxed = last_boxed(value)
    if boxed is not None:
        value = boxed.strip()
    if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return re.sub(r"^\*\*|\*\*$", "", value).strip()


def _canonical_unit(value: str) -> str:
    compact = re.sub(r"[^a-z0-9^]", "", value.lower())
    compact = compact.replace("squared", "^2").replace("cubed", "^3")
    return UNIT_ALIASES.get(compact, compact)


def _separate_terminal_unit(value: str) -> tuple[str, str | None]:
    value = _strip_math_wrappers(value).strip()
    degree = re.search(r"(?:\^\{?\\circ\}?|°)\s*$", value)
    if degree:
        return value[: degree.start()].strip(), "degree"
    percent = re.search(r"(?:\\%|%)\s*$", value)
    if percent:
        return value[: percent.start()].strip(), "percent"
    command_unit = re.search(
        r"\\(?:text|mathrm|mbox)\{([^{}]+)\}(\^\{?[23]\}?)?\s*$",
        value,
        flags=re.DOTALL,
    )
    if command_unit and re.fullmatch(r"[A-Za-z\s^0-9.:-]+", command_unit.group(1)):
        prefix = value[: command_unit.start()].strip()
        label = command_unit.group(1).strip().lower()
        scale_words = {"hundred", "thousand", "million", "billion", "trillion", "dozen"}
        if prefix and not any(word in label.split() for word in scale_words):
            unit = _canonical_unit(label)
            if command_unit.group(2):
                unit += "^" + re.search(r"[23]", command_unit.group(2)).group(0)
            return prefix, unit
    plain_unit = re.search(
        r"\s+(degrees?|dollars?|cents?|percent(?:age)?|square\s+units?|cubic\s+units?|"
        r"centimeters?|cm|mm|meters?|m|km|kilometers?|inches?|feet|foot|yards?|miles?|"
        r"seconds?|minutes?|hours?|units?)\s*$",
        value,
        flags=re.IGNORECASE,
    )
    if plain_unit:
        return value[: plain_unit.start()].strip(), _canonical_unit(plain_unit.group(1))
    return value, None


def _fix_fracs(value: str) -> str:
    pieces = value.split(r"\frac")
    rebuilt = pieces[0]
    for piece in pieces[1:]:
        rebuilt += r"\frac"
        if not piece:
            continue
        if piece[0] == "{":
            rebuilt += piece
            continue
        numerator = piece[0]
        remainder = piece[1:]
        if not remainder:
            rebuilt += "{" + numerator + "}"
        elif remainder[0] == "{":
            rebuilt += "{" + numerator + "}" + remainder
        else:
            rebuilt += "{" + numerator + "}{" + remainder[0] + "}" + remainder[1:]
    return rebuilt


def normalize_answer(value: str) -> str:
    """Conventional MATH normalization after removing a recognized unit."""
    value, _ = _separate_terminal_unit(value)
    value = value.strip().replace("\n", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\[", "").replace(r"\]", "")
    value = value.replace(r"\!", "").replace("\\\\", "\\")
    value = value.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    value = value.replace(r"\$", "").replace("$", "")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mbox\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"^(?:x|y|z|a|b|c)\s*=", "", value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        value = value.rstrip("0").rstrip(".")
    value = re.sub(r"\\sqrt([^\{])", r"\\sqrt{\1}", value)
    value = _fix_fracs(value)
    if value.startswith("."):
        value = "0" + value
    if re.fullmatch(r"-?\d+/\d+", value):
        numerator, denominator = value.split("/", 1)
        value = rf"\frac{{{numerator}}}{{{denominator}}}"
    if re.fullmatch(r"\([A-E]\)", value, flags=re.IGNORECASE):
        value = value[1:-1]
    return value


def _balanced_replace(command: str, text: str, unary: bool = False) -> str:
    marker = "\\" + command
    while marker in text:
        start = text.rfind(marker)
        first = text.find("{", start + len(marker))
        if first < 0:
            break

        def group_end(group_start: int, current_text: str = text) -> int:
            depth = 0
            for index in range(group_start, len(current_text)):
                if current_text[index] == "{":
                    depth += 1
                elif current_text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return index
            return -1

        first_end = group_end(first)
        if first_end < 0:
            break
        first_value = text[first + 1 : first_end]
        if unary:
            text = text[:start] + f"sqrt(({first_value}))" + text[first_end + 1 :]
            continue
        second = text.find("{", first_end + 1)
        if second != first_end + 1:
            break
        second_end = group_end(second)
        if second_end < 0:
            break
        second_value = text[second + 1 : second_end]
        text = text[:start] + f"(({first_value})/({second_value}))" + text[second_end + 1 :]
    return text


def _latex_to_sympy_text(value: str) -> str:
    value, _ = _separate_terminal_unit(value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    value = re.sub(r"\\sqrt\s*([^\s{}])", r"\\sqrt{\1}", value)
    value = _balanced_replace("frac", value)
    value = _balanced_replace("sqrt", value, unary=True)
    value = value.replace(r"\cdot", "*").replace(r"\times", "*")
    value = value.replace(r"\pi", "pi").replace(r"\infty", "oo")
    value = value.replace(r"\mathrm{i}", "I").replace(r"\imath", "I")
    value = re.sub(r"(?<=\d)i\b", "*I", value)
    value = re.sub(r"\bi\b", "I", value)
    value = value.replace("{", "(").replace("}", ")").replace("^", "**")
    return value.replace(r"\,", "").replace(r"\!", "").strip()


def _parse_scalar(value: str) -> sympy.Expr | None:
    text = _latex_to_sympy_text(value)
    if not text or any(token in text for token in (r"\begin", r"\cup", "[", "]")):
        return None
    if len(text) > 256 or re.search(r"\*\*\([^)]*\*\*", text) or text.count("=") > 1:
        return None
    try:
        return parse_expr(
            text,
            local_dict={"i": sympy.I, "I": sympy.I, "pi": sympy.pi, "oo": sympy.oo},
            transformations=PARSER_TRANSFORMATIONS,
            evaluate=False,
        )
    except (SyntaxError, TokenError, TypeError, ValueError, NameError, AttributeError):
        return None


def split_top_level(value: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    close_to_open = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(value):
        if char in depths:
            depths[char] += 1
        elif char in close_to_open:
            depths[close_to_open[char]] -= 1
        elif char == delimiter and all(depth == 0 for depth in depths.values()):
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _matrix_components(value: str) -> list[str] | None:
    match = re.fullmatch(
        r"\s*\\begin\{[pbvBV]?matrix\}(.*?)\\end\{[pbvBV]?matrix\}\s*",
        _strip_math_wrappers(value),
        flags=re.DOTALL,
    )
    if not match:
        return None
    return [cell.strip() for row in match.group(1).split(r"\\") for cell in row.split("&")]


def math_equivalent(prediction: str, gold: str) -> tuple[bool, str]:
    """Conservative deterministic equivalence with matrix, unit, and SymPy support."""
    prediction = _strip_math_wrappers(prediction)
    gold = _strip_math_wrappers(gold)
    if not prediction:
        return False, "missing"
    prediction_value, prediction_unit = _separate_terminal_unit(prediction)
    gold_value, gold_unit = _separate_terminal_unit(gold)
    if prediction_unit and gold_unit and prediction_unit != gold_unit:
        return False, "unit_mismatch"
    if normalize_answer(prediction_value) == normalize_answer(gold_value):
        return True, "normalized_exact_with_unit" if prediction_unit or gold_unit else "normalized_exact"

    prediction_matrix = _matrix_components(prediction_value)
    gold_matrix = _matrix_components(gold_value)
    if prediction_matrix is not None or gold_matrix is not None:
        if prediction_matrix is None or gold_matrix is None or len(prediction_matrix) != len(gold_matrix):
            return False, "matrix_shape_mismatch"
        matches = [math_equivalent(left, right)[0] for left, right in zip(prediction_matrix, gold_matrix)]
        return all(matches), "componentwise_matrix" if all(matches) else "matrix_mismatch"

    if (
        prediction_value[:1] in "({"
        and gold_value[:1] in "({"
        and prediction_value[-1:] in ")}"
        and gold_value[-1:] in ")}"
    ):
        prediction_parts = split_top_level(prediction_value[1:-1])
        gold_parts = split_top_level(gold_value[1:-1])
        if len(prediction_parts) == len(gold_parts) and len(prediction_parts) > 1:
            matches = [math_equivalent(left, right)[0] for left, right in zip(prediction_parts, gold_parts)]
            return all(matches), "componentwise_tuple" if all(matches) else "tuple_mismatch"

    if prediction_value.count("=") == 1 and gold_value.count("=") == 1:
        pred_left, pred_right = prediction_value.split("=", 1)
        gold_left, gold_right = gold_value.split("=", 1)
        values = tuple(_parse_scalar(value) for value in (pred_left, pred_right, gold_left, gold_right))
        if all(value is not None for value in values):
            pred_expression = sympy.expand(values[0] - values[1])
            gold_expression = sympy.expand(values[2] - values[3])
            if sympy.simplify(pred_expression - gold_expression) == 0:
                return True, "symbolic_equation"
            if gold_expression != 0:
                ratio = sympy.simplify(pred_expression / gold_expression)
                if not ratio.free_symbols and ratio != 0:
                    return True, "symbolic_equation_scaled"
        return False, "equation_mismatch"

    prediction_expression = _parse_scalar(prediction_value)
    gold_expression = _parse_scalar(gold_value)
    if prediction_expression is not None and gold_expression is not None:
        try:
            if sympy.simplify(prediction_expression - gold_expression) == 0:
                return True, "symbolic"
            difference = complex(sympy.N(prediction_expression - gold_expression, 15))
            if math.isfinite(abs(difference)) and abs(difference) <= 1e-9:
                return True, "numeric_tolerance"
        except (TypeError, ValueError):
            pass
        return False, "symbolic_mismatch"
    return False, "normalized_mismatch"


def evaluate_answer(prediction: str, gold: GoldSpec) -> EvaluationResult:
    if not gold.requires_multiple_answers:
        correct, method = math_equivalent(prediction, gold.canonical_answer)
        return EvaluationResult(correct, method)
    predicted_components = split_top_level(_strip_math_wrappers(prediction))
    if len(predicted_components) != len(gold.components):
        return EvaluationResult(False, "unordered_component_count_mismatch")
    unused = set(range(len(gold.components)))
    methods: list[str] = []
    for predicted in predicted_components:
        matched = next(
            (
                (index, method)
                for index in sorted(unused)
                for equivalent, method in [math_equivalent(predicted, gold.components[index])]
                if equivalent
            ),
            None,
        )
        if matched is None:
            return EvaluationResult(False, "unordered_component_mismatch", tuple(methods))
        unused.remove(matched[0])
        methods.append(matched[1])
    return EvaluationResult(True, "unordered_componentwise_math_equivalence", tuple(methods))


def answers_equal(prediction: str, gold: str) -> bool:
    """Backward-compatible scalar/matrix/unit mathematical equivalence."""
    return math_equivalent(prediction, gold)[0]


def gold_metadata(gold: GoldSpec) -> dict[str, object]:
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "boxed_answers": list(gold.boxed_answers),
        "answer_components": list(gold.components),
        "requires_multiple_answers": gold.requires_multiple_answers,
    }
