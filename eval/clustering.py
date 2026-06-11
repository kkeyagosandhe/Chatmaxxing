from google import genai
from langfuse import get_client
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from safe_call import safe_generate
import os
import json

load_dotenv()

def _default_client():
    return genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
langfuse = get_client()

def cluster_failures(results: list, n_clusters: int = 3, gemini_client=None) -> list:
    DETECTORS = ["goal_drift", "hallucination", "wrong_disposition"]

    failures = [r for r in results if r["any_failure"] or any(
        r[d].get("uncertain") for d in DETECTORS
    )]
    with langfuse.start_as_current_observation(as_type="span", name="clustering") as span:

        if not failures:
            print("No failures or uncertain cases to cluster.")
            span.update(output="[]")
            return []

        # Build failure descriptions
        descriptions = []
        for f in failures:
            parts = []
            if f["goal_drift"]["drifted"]:
                parts.append(f"Goal Drift: {f['goal_drift']['reason']}")
            if f["hallucination"]["hallucinated"]:
                parts.append(f"Hallucination: {f['hallucination']['reason']}")
            if f["wrong_disposition"]["wrong_disposition"]:
                parts.append(f"Wrong Disposition: {f['wrong_disposition']['reason']}")
            # uncertain-but-passed: surface the split detector's reasoning so the
            # case is clusterable instead of contributing an empty string
            if not parts:
                for d in DETECTORS:
                    if f[d].get("uncertain"):
                        parts.append(f"Uncertain ({d}): {f[d].get('reason', '')}")
            descriptions.append(" | ".join(parts) or "uncertain case")

        # Vectorize
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(descriptions)

        # Cluster
        n = min(n_clusters, len(failures))
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Group failures by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                "ticket_id": failures[i]["ticket_id"],
                "description": descriptions[i],
                "uncertain": any(failures[i][d].get("uncertain") for d in DETECTORS),
            })

        # Ask Gemini to name each cluster
        results = []
        for label, members in clusters.items():
            sample = members[0]["description"]
            prompt = f"""You are an AI eval analyst. Given these failure descriptions from a customer support agent, provide a root cause label in 5 words or less and a one sentence fix suggestion.

Sample failure: {sample}

Reply with JSON only:
{{"root_cause": "5 word label", "fix": "one sentence fix"}}"""

            _client = gemini_client or _default_client()
            response = safe_generate(_client, "gemini-2.5-flash-lite", prompt)

            import re
            text = response.text.strip()
            text = re.sub(r"```json|```", "", text).strip()
            label_data = json.loads(text)

            if isinstance(label_data, list):
                label_data = label_data[0]

            results.append({
                "cluster_id": int(label),
                "root_cause": label_data.get("root_cause", label_data.get("label", "Unknown cluster")),
                "fix": label_data.get("fix", label_data.get("suggestion", "No fix suggested")),
                "count": len(members),
                "tickets": [m["ticket_id"] for m in members],
                "has_uncertain": any(m["uncertain"] for m in members),
            })

        span.update(output=str(results))
        return results