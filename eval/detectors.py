from google import genai
from langfuse import get_client
from dotenv import load_dotenv
from safe_call import safe_generate
import os
import json
import re
from collections import Counter

# Response length bounds from 5th/95th percentile of the ticket dataset.
_RESPONSE_MIN_WORDS = 37
_RESPONSE_MAX_WORDS = 201
_VALID_DISPOSITIONS = {"RESOLVED", "ESCALATE", "NEED_MORE_INFO"}
# Confidence is on a 0.0-1.0 scale (see AgentResponse prompt). High confidence
# on a flagged ticket signals a pipeline inconsistency, not a confident correct
# answer — anything above this threshold is flagged.
_CONFIDENCE_FAILURE_THRESHOLD = 0.7


def validate_schema(response: str, disposition: str, confidence: float, any_failure: bool) -> dict:
    """
    Deterministic cross-field checks. No LLM calls. Returns a dict of
    violations found (empty dict = clean).
    """
    violations = {}

    # 1. Confidence-failure agreement: high confidence on a flagged ticket is
    #    a pipeline red flag, not evidence the response was correct.
    if any_failure and confidence > _CONFIDENCE_FAILURE_THRESHOLD:
        violations["confidence_failure_agreement"] = (
            f"confidence={confidence} but ticket has failures — "
            "high confidence on a flagged response signals a pipeline inconsistency"
        )

    # 2. Escalation coherence: an ESCALATE response should not assert the issue
    #    is already resolved, and a RESOLVED response should not say it is being
    #    handed off. Phrases (not bare words) to avoid benign false positives
    #    like "once the specialist is done" or "feel free to follow up".
    response_lower = response.lower()
    if disposition == "ESCALATE" and re.search(
        r"\b(issue is (now )?resolved|has been (resolved|fixed|solved)|problem is (now )?fixed|this is resolved)\b",
        response_lower,
    ):
        violations["escalation_coherence"] = (
            "disposition is ESCALATE but response asserts the issue is already resolved"
        )
    if disposition == "RESOLVED" and re.search(
        r"\b(escalating (this|your)|routing (this|you) to|a specialist will|our team will (look|investigate|follow up))\b",
        response_lower,
    ):
        violations["escalation_coherence"] = (
            "disposition is RESOLVED but response says it is being escalated/handed off"
        )

    # 3. Response length bounds (data-driven: 5th/95th percentile).
    word_count = len(response.split())
    if word_count < _RESPONSE_MIN_WORDS and disposition != "ESCALATE":
        violations["response_too_short"] = (
            f"{word_count} words — below floor of {_RESPONSE_MIN_WORDS} "
            "for a non-escalation response"
        )
    if word_count > _RESPONSE_MAX_WORDS:
        violations["response_too_long"] = (
            f"{word_count} words — above ceiling of {_RESPONSE_MAX_WORDS}"
        )

    # 4. Disposition must be from the closed enum (Pydantic enforces this at
    #    parse time, but re-check here so the eval surface is self-contained).
    if disposition not in _VALID_DISPOSITIONS:
        violations["invalid_disposition"] = (
            f"'{disposition}' is not a valid disposition"
        )

    return {
        "violations": violations,
        "passed": len(violations) == 0,
        "violation_count": len(violations),
    }

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
langfuse = get_client()

N_VOTES = 3


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def _majority_vote(samples: list, flag_key: str) -> dict:
    flags = [bool(s[flag_key]) for s in samples]
    majority_flag = Counter(flags).most_common(1)[0][0]
    reason = next(
        (s.get("reason", "") for s in samples if bool(s[flag_key]) == majority_flag),
        "",
    )
    scores = [s.get("score") for s in samples if isinstance(s.get("score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 3) if scores else None
    return {
        "score": avg_score,
        "reason": reason,
        flag_key: majority_flag,
        "votes": f"{flags.count(majority_flag)}/{len(flags)}",
        "uncertain": flags.count(True) == 1 or flags.count(False) == 1,
    }


def _vote(prompt: str, flag_key: str) -> dict:
    samples = []
    for _ in range(N_VOTES):
        result = safe_generate(client, "gemini-2.5-flash", prompt)
        samples.append(_parse_json(result.text))
    return _majority_vote(samples, flag_key)


def detect_goal_drift(query: str, response: str) -> dict:
    prompt = f"""You are an evaluator checking if an agent's response addresses the customer's question.

Customer Query: {query}
Agent Response: {response}

RUBRIC — A response ADDRESSES the question if:
- It directly answers the stated question, OR
- It acknowledges the question and explains why more information is needed, OR
- It escalates with a clear reason related to the question

A response DRIFTS if:
- It talks about something the customer never mentioned, OR
- It ignores the customer's actual question entirely and answers a different question

Apply the rubric above. Reply with JSON only:
{{"score": 0.0 to 1.0, "reason": "one sentence citing which rubric case applies", "drifted": true or false}}"""
    return _vote(prompt, "drifted")


def detect_hallucination(query: str, response: str, context: dict) -> dict:
    prompt = f"""You are an evaluator checking if a customer support agent hallucinated.

The agent was given this full context:
- Customer Name: {context['customer_name']}
- Product: {context['product']}
- Issue Type: {context['ticket_type']}
- Subject: {context['subject']}
- Description: {query}

The agent responded with:
{response}

RUBRIC — Flag as hallucination ONLY if the agent introduces:
- Specific facts (URLs, phone numbers, reference IDs, version numbers, policy details) NOT present in the context
- Claims about what prior agents said or did when no such history is in the context
- Product features or troubleshooting steps specific to a different product

Do NOT flag as hallucination:
- Using the customer's name, product, or issue type from the context
- Generic support advice (restart device, check settings, contact support)
- Standard troubleshooting that applies broadly to the product category

Apply the rubric above. Reply with JSON only:
{{"score": 0.0 to 1.0, "reason": "one sentence citing what was hallucinated or why it passed", "hallucinated": true or false}}"""
    return _vote(prompt, "hallucinated")


def detect_wrong_disposition(response: str, ticket_type: str) -> dict:
    prompt = f"""You are an evaluator checking if the agent chose the correct disposition.

Ticket Type: {ticket_type}
Agent Response (last 200 chars): {response[-200:]}

RUBRIC for disposition:
- [RESOLVED]: Use if the agent provided a complete answer or solution that fully addresses the issue
- [ESCALATE]: Use if the issue requires human review, specialized access, or is beyond agent scope
- [NEED_MORE_INFO]: Use if critical details are missing and the agent cannot proceed without them

WRONG disposition examples:
- Picked [RESOLVED] but only acknowledged the issue without solving it
- Picked [NEED_MORE_INFO] when the ticket already contains enough detail to attempt help or escalate
- Picked [ESCALATE] for a simple question the agent could have answered

Apply the rubric above. Reply with JSON only:
{{"score": 0.0 to 1.0, "reason": "one sentence citing which rubric applies", "wrong_disposition": true or false}}"""
    return _vote(prompt, "wrong_disposition")


def run_all_detectors(
    ticket_id: int,
    query: str,
    response: str,
    ticket_type: str,
    context: dict,
    disposition: str = "UNKNOWN",
    confidence: float = 0.0,
) -> dict:
    with langfuse.start_as_current_observation(as_type="span", name="eval-run") as span:

        goal_drift = detect_goal_drift(query, response)
        hallucination = detect_hallucination(query, response, context)
        wrong_disposition = detect_wrong_disposition(response, ticket_type)

        any_failure = any([
            goal_drift["drifted"],
            hallucination["hallucinated"],
            wrong_disposition["wrong_disposition"],
        ])

        schema = validate_schema(response, disposition, confidence, any_failure)

        results = {
            "ticket_id": ticket_id,
            "goal_drift": goal_drift,
            "hallucination": hallucination,
            "wrong_disposition": wrong_disposition,
            "schema": schema,
            "any_failure": any_failure or not schema["passed"],
        }

        span.update(
            input=query,
            output=str(results)
        )

        return results