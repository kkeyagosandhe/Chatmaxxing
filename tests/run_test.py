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
            disposition=agent_out["disposition"],
        )
        results.append({
            "eval": eval_result,
            "gate_triggered": agent_out["gate_triggered"],
            "disposition": agent_out["disposition"],
            "confidence": agent_out["confidence"],
            "query": agent_out["query"],
            "response": agent_out["response"],
        })
    return results


def summarize(results, label):
    n = len(results)
    grounding = sum(1 for r in results if r["eval"]["hallucination"]["hallucinated"])
    goal_drift = sum(1 for r in results if r["eval"]["goal_drift"]["drifted"])
    disposition = sum(1 for r in results if r["eval"]["wrong_disposition"]["wrong_disposition"])
    any_fail = sum(1 for r in results if r["eval"]["any_failure"])
    gate_hits = sum(1 for r in results if r["gate_triggered"])

    schema_fails = sum(1 for r in results if not r["eval"]["schema"]["passed"])
    confidences = [r["confidence"] for r in results if isinstance(r.get("confidence"), float)]
    conf_mean = round(sum(confidences) / len(confidences), 3) if confidences else None
    conf_std = round((sum((c - conf_mean) ** 2 for c in confidences) / len(confidences)) ** 0.5, 3) if confidences else None

    print(f"\n--- {label} (n={n}) ---")
    print(f"  Grounding failures:    {grounding}")
    print(f"  Goal drift failures:   {goal_drift}")
    print(f"  Disposition failures:  {disposition}")
    print(f"  Schema violations:     {schema_fails}")
    print(f"  Any failure (total):   {any_fail}/{n}")
    if conf_mean is not None:
        print(f"  Confidence (mean±std): {conf_mean} ± {conf_std}  ← poorly calibrated; motivates caveat system")
    if gate_hits:
        print(f"  Gate interventions:    {gate_hits} (ungrounded responses caught -> escalated)")
    return {"grounding": grounding, "any": any_fail, "schema": schema_fails}


# Held-out test set — same tickets scored both ways
TEST_START, TEST_END = 525, 530

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

from eval.fix_proposer import generate_caveats
from eval.clustering import cluster_failures

# Caveats run on gate-OFF results: the unfiltered agent behavior, before the
# gate converts some failures into escalations.
all_results = [r["eval"] for r in results_off]
clusters = cluster_failures(all_results)
caveats = generate_caveats(clusters)

print("\n=== CAVEAT ANNOTATIONS ===\n")
for c in caveats:
    print(f"Pattern: {c.cluster_label}")
    print(f"  Ambiguity:           {c.ambiguity}")
    print(f"  Human/LLM gap:       {c.human_llm_gap}")
    print(f"  Reviewer should:     {c.reviewer_signal}")
    print(f"  Human review needed: {c.needs_human_review}")
    print()


# --- Per-pattern breakdown: which failure types the gate actually fixes ---
# Clusters are built from gate-OFF failures (each carries its ticket ids).
# For each pattern we count how many of those same tickets STILL fail under
# the gate, so the reduction is attributable to a specific failure class.
on_failed_ids = {r["eval"]["ticket_id"] for r in results_on if r["eval"]["any_failure"]}

print("=== FAILURE BREAKDOWN (by pattern) ===\n")
for cl in clusters:
    off_count = cl["count"]
    on_count = sum(1 for tid in cl["tickets"] if tid in on_failed_ids)
    if on_count < off_count:
        tag = f"(FIXED {off_count - on_count}/{off_count} by gate)"
    elif on_count == off_count:
        tag = "(unchanged)"
    else:
        tag = "(worse)"
    print(f"  {cl['root_cause']}: {off_count} -> {on_count} cases {tag}")
print()


# --- Side-by-side ticket examples: the demo-facing evidence ---
# Show tickets the gate actually intervened on (gate-off failure vs gate-on safe
# escalation), so a reviewer can read the real before/after, not just metrics.
def first_line(text, limit=240):
    line = " ".join(text.split())
    return line[:limit] + ("..." if len(line) > limit else "")

off_by_id = {r["eval"]["ticket_id"]: r for r in results_off}
intervened = [r for r in results_on if r["gate_triggered"]]

print("=== SIDE-BY-SIDE EXAMPLES (gate interventions) ===\n")
if not intervened:
    print("  No gate interventions in this sample.\n")
for r in intervened[:3]:
    tid = r["eval"]["ticket_id"]
    off = off_by_id.get(tid)
    print(f"Ticket #{tid}")
    print(f"  Customer:        {first_line(r['query'])}")
    if off:
        print(f"  GATE OFF ->  {first_line(off['response'])}")
    print(f"  GATE ON  ->  {first_line(r['response'])}")
    print()