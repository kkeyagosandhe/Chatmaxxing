"""
Deterministic grounding gate (Phase 4 / structural fix).

After the agent generates a response, this checks — with NO LLM, no API calls,
pure Python — whether the response asserts concrete facts that do NOT appear in
the conversation context. If it does, the claim is ungrounded.

This is a structural guarantee, like the Pydantic schema: an ungrounded URL or
fabricated reference number CANNOT pass the gate, so it can never reach the
customer. The cost is zero (local string ops) and the result is fully
deterministic (same input -> same verdict, no noise).

Scope (honest limitation): catches concrete, checkable claims — URLs, numeric
tokens like order/reference numbers, and capitalised multi-word entities. It does
NOT catch subtle semantic hallucination (e.g. promising "prioritisation" that was
never offered). That is noted as future work (hybrid deterministic + LLM check).
"""

import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_claims(response_text: str) -> dict:
    """Pull concrete, checkable claims out of the agent response."""
    urls = re.findall(r"https?://\S+|www\.\S+", response_text)

    # Numeric tokens of length >= 4 (order numbers, reference IDs, phone numbers)
    numbers = re.findall(r"\b\d{4,}\b", response_text)

    # Capitalised multi-word phrases (likely proper nouns / product names),
    # ignoring sentence-start single words to reduce false positives.
    entities = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", response_text)

    return {"urls": urls, "numbers": numbers, "entities": entities}


def check_grounding(response_text: str, context_text: str) -> dict:
    """
    Returns {grounded: bool, violations: [...]}.
    A claim is a violation if it appears in the response but NOT in the context.
    """
    ctx = _normalize(context_text)
    claims = extract_claims(response_text)
    violations = []

    for url in claims["urls"]:
        # strip trailing punctuation that regex may capture
        clean = url.rstrip(".,!?)")
        if clean.lower() not in ctx:
            violations.append(f"URL not in context: {clean}")

    for num in claims["numbers"]:
        if num not in ctx:
            violations.append(f"Number not in context: {num}")

    for ent in claims["entities"]:
        if _normalize(ent) not in ctx:
            violations.append(f"Entity not in context: {ent}")

    return {"grounded": len(violations) == 0, "violations": violations}