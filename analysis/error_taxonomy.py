"""
Defines a structured taxonomy for categorizing LLM reasoning failures.
"""

from enum import Enum


class ReasoningError(Enum):
    ARITHMETIC_ERROR = "Incorrect numerical computation"
    LOGICAL_LEAP = "Conclusion without sufficient justification"
    MISSING_PREMISE = "Omission of a required assumption"
    CONTRADICTION = "Internal inconsistency in reasoning"
    MISINTERPRETATION = "Incorrect understanding of the prompt"
    OVERGENERALIZATION = "Invalid general rule inferred from limited data"


def classify_error(reasoning_text: str) -> list:
    """
    Placeholder heuristic-based classifier.
    Intended to be extended with human annotation or ML models.
    """
    errors = []

    if "therefore" in reasoning_text and "because" not in reasoning_text:
        errors.append(ReasoningError.LOGICAL_LEAP)

    if "2 + 2 = 5" in reasoning_text:
        errors.append(ReasoningError.ARITHMETIC_ERROR)

    return errors
