from google import genai
from google.genai import types
from langfuse import get_client
from dotenv import load_dotenv
from safe_call import safe_generate_structured
from grounding_gate import check_grounding
from pydantic import BaseModel
from typing import Literal
import os
import re
import pandas as pd

load_dotenv()

client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-central1"
)
langfuse = get_client()


class AgentResponse(BaseModel):
    response_text: str
    disposition: Literal["RESOLVED", "ESCALATE", "NEED_MORE_INFO"]
    confidence: float
    reasoning: str


DEFAULT_PROMPT = """You are a customer support agent.

Product: {product}
Issue Type: {ticket_type}
Subject: {subject}
Description: {description}

Respond to the customer.
Set 'disposition' to RESOLVED, ESCALATE, or NEED_MORE_INFO.
Set 'confidence' between 0.0 and 1.0.
Set 'reasoning' to a one-sentence justification."""


# Safe fallback message used when the grounding gate rejects a response.
GROUNDING_FALLBACK = (
    "Thanks for reaching out. I want to make sure I give you accurate "
    "information, so I'm routing this to a specialist who can look into the "
    "details and follow up with you directly."
)


def load_tickets(path="data/tickets_clean.csv"):
    return pd.read_csv(path)


def format_ticket(row: dict) -> dict:
    """Normalizes any ticket schema into a standard shape the agent expects."""
    return {
        "ticket_id": row.get("Ticket ID") or row.get("tweet_id") or row.get("id"),
        "description": row.get("Ticket Description") or row.get("text") or row.get("message") or "",
        "product": row.get("Product Purchased") or row.get("brand") or "Unknown",
        "ticket_type": row.get("Ticket Type") or row.get("category") or "General",
        "subject": row.get("Ticket Subject") or row.get("subject") or "",
        "customer_name": row.get("Customer Name") or row.get("author") or "Customer",
        "satisfaction": row.get("Customer Satisfaction Rating") or row.get("rating"),
    }


def run_agent(row: dict, prompt_template: str = DEFAULT_PROMPT, use_grounding_gate: bool = False) -> dict:
    with langfuse.start_as_current_observation(as_type="span", name="agent-run") as span:

        ticket = format_ticket(row)

        prompt = prompt_template.format(
            product=ticket["product"],
            ticket_type=ticket["ticket_type"],
            subject=ticket["subject"],
            description=ticket["description"],
        )

        parsed = safe_generate_structured(client, "gemini-2.5-flash", prompt, AgentResponse)

        # Strip any tag the model wrote into the text, keep clean body.
        clean_text = re.sub(r'\[(RESOLVED|ESCALATE|NEED_MORE_INFO)\]', '', parsed.response_text).strip()
        disposition = parsed.disposition

        # --- Structural grounding gate ---
        gate_triggered = False
        gate_violations = []
        if use_grounding_gate:
            gate = check_grounding(clean_text, ticket["description"])
            if not gate["grounded"]:
                # Reject the ungrounded response, fall back to a safe escalation.
                gate_triggered = True
                gate_violations = gate["violations"]
                clean_text = GROUNDING_FALLBACK
                disposition = "ESCALATE"

        answer = f"{clean_text}\n[{disposition}]"

        span.update(
            input=ticket["description"],
            output=answer,
        )

        return {
            "ticket_id": ticket["ticket_id"],
            "query": ticket["description"],
            "response": answer,
            "ticket_type": ticket["ticket_type"],
            "context": ticket,
            "actual_satisfaction": ticket["satisfaction"],
            "disposition": disposition,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
            "gate_triggered": gate_triggered,
            "gate_violations": gate_violations,
        }