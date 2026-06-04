import sys
sys.path.append(".")
from agent.agent import load_tickets, run_agent
from eval.detectors import run_all_detectors

df_raw = load_tickets("data/twitter_clean.csv")

df = df_raw[df_raw['Ticket Description'].notna()].reset_index(drop=True)
df = df[df['Ticket Description'].str.len() > 100].reset_index(drop=True)
print(f"Clean conversations available: {len(df)} (removed {len(df_raw) - len(df)} short/empty rows)")


def run_eval(use_grounding_gate=False, start=0, end=10):
    """Run agent + detectors over a ticket range. Returns the list of per-ticket results."""
    results = []
    for i in range(start, end):
        row = df.iloc[i].to_dict()
        agent_out = run_agent(row, use_grounding_gate=use_grounding_gate)
        eval_result = run_all_detectors(
            ticket_id=agent_out["ticket_id"],
            query=agent_out["query"],
            response=agent_out["response"],
            ticket_type=agent_out["ticket_type"],
            context=agent_out["context"],
        )
        results.append({
            "eval": eval_result,
            "gate_triggered": agent_out["gate_triggered"],
            "disposition": agent_out["disposition"],
        })
    return results


def summarize(results, label):
    n = len(results)
    grounding = sum(1 for r in results if r["eval"]["hallucination"]["hallucinated"])
    goal_drift = sum(1 for r in results if r["eval"]["goal_drift"]["drifted"])
    disposition = sum(1 for r in results if r["eval"]["wrong_disposition"]["wrong_disposition"])
    any_fail = sum(1 for r in results if r["eval"]["any_failure"])
    gate_hits = sum(1 for r in results if r["gate_triggered"])

    print(f"\n--- {label} (n={n}) ---")
    print(f"  Grounding failures:    {grounding}")
    print(f"  Goal drift failures:   {goal_drift}")
    print(f"  Disposition failures:  {disposition}")
    print(f"  Any failure (total):   {any_fail}/{n}")
    if gate_hits:
        print(f"  Gate interventions:    {gate_hits} (ungrounded responses caught -> escalated)")
    return {"grounding": grounding, "any": any_fail}


# Held-out test set — same tickets scored both ways
TEST_START, TEST_END = 425, 445

print("\n=== EXPERIMENT: grounding gate OFF vs ON (same held-out tickets) ===")

print("\nRunning WITHOUT grounding gate...")
results_off = run_eval(use_grounding_gate=False, start=TEST_START, end=TEST_END)
summary_off = summarize(results_off, "GATE OFF")

print("\nRunning WITH grounding gate...")
results_on = run_eval(use_grounding_gate=True, start=TEST_START, end=TEST_END)
summary_on = summarize(results_on, "GATE ON")

# --- Headline result ---
print("\n=== RESULT ===")
print(f"Grounding failures:  {summary_off['grounding']} (gate off)  ->  {summary_on['grounding']} (gate on)")
print(f"Total failures:      {summary_off['any']} (gate off)  ->  {summary_on['any']} (gate on)")
print("\nNote: the gate converts ungrounded (confidently wrong) responses into")
print("safe escalations. Grounding-class failures are reduced structurally;")
print("total count reflects that dangerous failures become safe hand-offs.")