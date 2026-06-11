from google import genai
from langfuse import get_client
from dotenv import load_dotenv
from safe_call import safe_generate
from pydantic import BaseModel
import os
import json
import re

load_dotenv()

client = genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
langfuse = get_client()


class CaveatAnnotation(BaseModel):
    cluster_label: str
    ambiguity: str
    human_llm_gap: str
    reviewer_signal: str
    needs_human_review: bool


def generate_caveats(cluster_summaries: list, gemini_client=None) -> list[CaveatAnnotation]:
    """
    Takes cluster names + detector results, returns caveat annotations
    instead of prompt patches. Surfaces ambiguity for human review.
    """
    with langfuse.start_as_current_observation(as_type="span", name="caveat-annotator") as span:

        summary_lines = "\n".join([
            f"- {c['root_cause']} (affects {c['count']} tickets, "
            f"contains 2/3 split votes: {bool(c.get('has_uncertain'))})"
            for c in cluster_summaries
        ])

        prompt = f"""You are reviewing cases where an AI customer support detector flagged uncertainty.

Failure clusters identified (each line states whether the cluster contains 2/3 split votes):
{summary_lines}

For each cluster, return a JSON object with:
- cluster_label: short name for this failure pattern
- ambiguity: what makes this case genuinely grey — not clearly right or wrong
- human_llm_gap: why a human reviewer and an LLM evaluator might reach different conclusions here
- reviewer_signal: one concrete thing a human should look for when reviewing this
- needs_human_review: set this to exactly the "contains 2/3 split votes" value given for that cluster above

Do NOT suggest prompt rules. Do NOT suggest fixes.
Your job is to surface uncertainty, not eliminate it.

Return a JSON array of caveat objects, one per cluster, in the same order."""

        _client = gemini_client or client
        response = safe_generate(_client, "gemini-2.5-flash-lite", prompt)

        text = response.text.strip()
        text = re.sub(r"```json|```", "", text).strip()
        raw = json.loads(text)

        # The split-vote flag is ground truth from the detectors — overwrite
        # whatever the model returned so needs_human_review can't be hallucinated.
        caveats = []
        for item, cluster in zip(raw, cluster_summaries):
            item["needs_human_review"] = bool(cluster.get("has_uncertain"))
            caveats.append(CaveatAnnotation(**item))
        span.update(output=str([c.model_dump() for c in caveats]))
        return caveats
