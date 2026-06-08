"""
Compute Cohen's kappa between human labels and detector verdicts.

1. Load the human-labeled gold set
2. Search tickets 425-445 to find exact ticket_id matches
3. Run detectors on those tickets
4. Compare human labels vs detector verdicts
5. Compute kappa per detector
"""

import sys
sys.path.append(".")
import pandas as pd
from agent.agent import load_tickets, run_agent, format_ticket
from eval.detectors import run_all_detectors
from sklearn.metrics import cohen_kappa_score
import openpyxl
import math

# Load human labels
print("Loading human gold set...")
wb = openpyxl.load_workbook("gold_set_template.xlsx")
ws = wb.active

human_labels = []
for row in ws.iter_rows(min_row=2, values_only=True):
    human_labels.append({
        "current_message": row[1],
        "agent_response": row[2] if row[2] else None,  # use synthetic response if provided
        "goal_drift": str(row[3]).upper() == "TRUE",
        "hallucination": str(row[4]).upper() == "TRUE",
        "wrong_disposition": str(row[5]).upper() == "TRUE",
    })

print(f"Loaded {len(human_labels)} human-labeled tickets")

# Load Twitter dataset and search for matches in range 425-445
df = load_tickets("data/twitter_clean.csv")
df_search = df.iloc[425:446]  # tickets 425-445

matched = []
for human in human_labels:
    # Find the row where CURRENT MESSAGE matches
    # Extract just the customer's message from the full description
    for idx, row in df_search.iterrows():
        desc = row["Ticket Description"]
        if "CURRENT MESSAGE:" in desc:
            current_part = desc.split("CURRENT MESSAGE:")[-1].strip()
            # Match first 100 chars (handles truncation)
            if human["current_message"][:100] in current_part:
                matched.append({
                    "ticket_id": row["Ticket ID"],
                    "row": row.to_dict(),
                    "human": human,
                })
                break

print(f"Matched {len(matched)}/{len(human_labels)} tickets in range 425-445")

if len(matched) < len(human_labels):
    print(f"⚠️ Could not find all tickets. Proceeding with {len(matched)} matches.")

# Run detectors on matched tickets
print("\nRunning detectors on gold set (this may take a few minutes with 3-vote)...")
detector_verdicts = []

for i, item in enumerate(matched, 1):
    print(f"  {i}/{len(matched)}", end="\r")
    synthetic = item["human"].get("agent_response")
    if synthetic:
        # Gold set provides a known-bad response — test detectors directly on it
        # without running the live agent, so the synthetic failure is what gets evaluated.
        ticket = format_ticket(item["row"])
        result = {
            "ticket_id": ticket["ticket_id"],
            "query": ticket["description"],
            "response": synthetic,
            "ticket_type": ticket["ticket_type"],
            "context": ticket,
        }
    else:
        result = run_agent(item["row"], use_grounding_gate=False)
    eval_result = run_all_detectors(
        ticket_id=result["ticket_id"],
        query=result["query"],
        response=result["response"],
        ticket_type=result["ticket_type"],
        context=result["context"],
    )
    detector_verdicts.append(eval_result)

print()

# Build parallel arrays for kappa computation
human_goal = [m["human"]["goal_drift"] for m in matched]
human_hall = [m["human"]["hallucination"] for m in matched]
human_disp = [m["human"]["wrong_disposition"] for m in matched]

detector_goal = [v["goal_drift"]["drifted"] for v in detector_verdicts]
detector_hall = [v["hallucination"]["hallucinated"] for v in detector_verdicts]
detector_disp = [v["wrong_disposition"]["wrong_disposition"] for v in detector_verdicts]

# Compute Cohen's kappa
kappa_goal = cohen_kappa_score(human_goal, detector_goal)
kappa_hall = cohen_kappa_score(human_hall, detector_hall)
kappa_disp = cohen_kappa_score(human_disp, detector_disp)

# Raw agreement for context
agree_goal = sum(h == d for h, d in zip(human_goal, detector_goal)) / len(matched)
agree_hall = sum(h == d for h, d in zip(human_hall, detector_hall)) / len(matched)
agree_disp = sum(h == d for h, d in zip(human_disp, detector_disp)) / len(matched)

def interpret_kappa(k):
    if math.isnan(k):
        return "undefined (not enough variation)"
    if k < 0.20:
        return "slight agreement"
    elif k < 0.40:
        return "fair agreement"
    elif k < 0.60:
        return "moderate agreement"
    elif k < 0.80:
        return "substantial agreement"
    else:
        return "almost perfect agreement"

print("\n=== DETECTOR CALIBRATION (Cohen's Kappa) ===\n")

print(f"Goal Drift detector:")
print(f"  κ = {kappa_goal:.3f}  ({interpret_kappa(kappa_goal)})")
print(f"  Raw agreement: {agree_goal:.1%}\n")

print(f"Hallucination detector:")
print(f"  κ = {kappa_hall:.3f}  ({interpret_kappa(kappa_hall)})")
print(f"  Raw agreement: {agree_hall:.1%}\n")

print(f"Disposition detector:")
print(f"  κ = {kappa_disp:.3f}  ({interpret_kappa(kappa_disp)})")
print(f"  Raw agreement: {agree_disp:.1%}\n")

print("Interpretation (Landis & Koch 1977):")
print("  < 0.20 = slight, 0.21 to 0.40 = fair, 0.41 to 0.60 = moderate")
print("  0.61 to 0.80 = substantial, 0.81 to 1.00 = almost perfect")
print("\nTarget: κ ≥ 0.61 (substantial agreement)")