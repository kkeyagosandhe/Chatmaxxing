from google import genai
from langfuse import get_client
from dotenv import load_dotenv
from safe_call import safe_generate
import os
import json
import re

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
langfuse = get_client()

def propose_fix(clusters: list) -> str:
    with langfuse.start_as_current_observation(as_type="span", name="fix-proposer") as span:

        cluster_summary = "\n".join([
            f"- {c['root_cause']} (affects {c['count']} tickets): {c['fix']}"
            for c in clusters
        ])

        prompt = f"""You are an AI agent prompt engineer.

A customer support agent has the following recurring failure patterns:

{cluster_summary}

The current agent prompt is:
\"\"\"
You are a customer support agent.

Product: {{product}}
Issue Type: {{ticket_type}}
Subject: {{subject}}
Description: {{description}}

STRICT RULES:
- Do NOT use any customer name unless explicitly stated in the Description
- Do NOT assume the issue type beyond what is stated in the Description
- Do NOT introduce any information not present in the Description
- Only respond based on what the customer explicitly wrote

Provide a clear, helpful response to resolve this customer's issue.
End your response with exactly one of: [RESOLVED] [ESCALATE] [NEED_MORE_INFO]
\"\"\"

Return the current prompt above with ONE new rule added to the STRICT RULES section.
The rule must directly address this failure pattern: {clusters[0]['root_cause']} — {clusters[0]['fix']}
Do not rewrite, reorder, or remove anything else.
Return only the improved prompt text, no explanation, no code blocks."""

        response = safe_generate(client, "gemini-2.5-flash", prompt)

        improved_prompt = response.text.strip()
        span.update(output=improved_prompt)
        return improved_prompt