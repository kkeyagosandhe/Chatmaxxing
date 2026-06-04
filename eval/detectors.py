from google import genai
from langfuse import get_client
from dotenv import load_dotenv
from safe_call import safe_generate
import os
import json
import re
from collections import Counter

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


def run_all_detectors(ticket_id: int, query: str, response: str, ticket_type: str, context: dict) -> dict:
    with langfuse.start_as_current_observation(as_type="span", name="eval-run") as span:

        goal_drift = detect_goal_drift(query, response)
        hallucination = detect_hallucination(query, response, context)
        disposition = detect_wrong_disposition(response, ticket_type)

        results = {
            "ticket_id": ticket_id,
            "goal_drift": goal_drift,
            "hallucination": hallucination,
            "wrong_disposition": disposition,
            "any_failure": any([
                goal_drift["drifted"],
                hallucination["hallucinated"],
                disposition["wrong_disposition"]
            ])
        }

        span.update(
            input=query,
            output=str(results)
        )

        return results