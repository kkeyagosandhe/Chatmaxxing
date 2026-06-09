"""
Chatmaxxing MCP Server

Exposes the eval pipeline as tools that any MCP-compatible client
(Claude Desktop, Claude Code, etc.) can call directly.

Setup:
  1. Add your GEMINI_API_KEY to .env
  2. Run: uv run python mcp_server.py

Tools exposed:
  - run_eval          : run the full eval pipeline on a ticket range
  - get_ticket_detail : get full drill-down for a single ticket id
"""

import sys
sys.path.append(".")

import json
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Chatmaxxing Eval")

# In-memory cache so get_ticket_detail can look up results from the last run
_last_run: dict = {}


@mcp.tool()
def run_eval(start: int = 425, end: int = 435) -> str:
    """
    Run the AI agent quality eval pipeline on a range of support tickets.

    Evaluates each ticket for:
    - Factual accuracy (did the agent introduce false information?)
    - Goal drift (did the agent stay on topic?)
    - Correct outcome (did the agent choose the right resolution type?)
    - Format compliance (did the response meet structural rules?)

    Also clusters failures into patterns and generates caveat annotations
    that explain why ambiguous cases are hard to judge automatically.

    Args:
        start: Index of the first ticket to evaluate (default 425)
        end:   Index after the last ticket to evaluate (default 435)

    Returns:
        A summary of findings including per-ticket results and failure patterns.
    """
    from agent.agent import load_tickets, run_agent
    from eval.detectors import run_all_detectors
    from eval.clustering import cluster_failures
    from eval.fix_proposer import generate_caveats

    df_raw = load_tickets("data/twitter_clean.csv")
    df = df_raw[df_raw["Ticket Description"].notna()].reset_index(drop=True)
    df = df[df["Ticket Description"].str.len() > 100].reset_index(drop=True)

    results = []
    total = end - start

    for i in range(start, end):
        row = df.iloc[i].to_dict()
        agent_out = run_agent(row, use_grounding_gate=False)
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
            "disposition": agent_out["disposition"],
            "confidence": agent_out["confidence"],
            "query": agent_out["query"],
            "response": agent_out["response"],
        })

    all_evals = [r["eval"] for r in results]
    clusters = cluster_failures(all_evals)
    caveats = generate_caveats(clusters) if clusters else []

    # Cache for get_ticket_detail
    _last_run["results"] = results
    _last_run["clusters"] = clusters
    _last_run["caveats"] = caveats

    # Build caveat lookup
    caveat_map = {}
    for i, cl in enumerate(clusters):
        if i < len(caveats):
            for tid in cl.get("tickets", []):
                caveat_map[tid] = caveats[i]

    # Summary
    n = len(results)
    grounding = sum(1 for r in results if r["eval"]["hallucination"]["hallucinated"])
    disposition_err = sum(1 for r in results if r["eval"]["wrong_disposition"]["wrong_disposition"])
    goal_drift = sum(1 for r in results if r["eval"]["goal_drift"]["drifted"])
    schema_fails = sum(1 for r in results if not r["eval"]["schema"]["passed"])
    any_fail = sum(1 for r in results if r["eval"]["any_failure"])
    needs_review = sum(1 for c in caveat_map.values() if getattr(c, "needs_human_review", False))

    confidences = [r["confidence"] for r in results if isinstance(r.get("confidence"), float)]
    conf_mean = round(sum(confidences) / len(confidences), 2) if confidences else None

    ticket_lines = []
    for r in results:
        ev = r["eval"]
        tid = ev["ticket_id"]
        issues = []
        if ev["hallucination"]["hallucinated"]:
            issues.append("inaccurate response")
        if ev["goal_drift"]["drifted"]:
            issues.append("off topic")
        if ev["wrong_disposition"]["wrong_disposition"]:
            issues.append("wrong outcome")
        if not ev["schema"]["passed"]:
            issues.append("format issue")
        status = "ISSUES: " + ", ".join(issues) if issues else "clean"
        caveat = caveat_map.get(tid)
        review = " [needs human review]" if caveat and getattr(caveat, "needs_human_review", False) else ""
        ticket_lines.append(f"  Ticket #{tid} ({r['disposition']}) — {status}{review}")

    pattern_lines = []
    for i, cl in enumerate(clusters):
        if i < len(caveats):
            c = caveats[i]
            pattern_lines.append(
                f"  Pattern: {c.cluster_label}\n"
                f"    Why it's ambiguous: {c.ambiguity}\n"
                f"    Reviewer should check: {c.reviewer_signal}\n"
                f"    Human review needed: {'Yes' if c.needs_human_review else 'No'}"
            )

    output = f"""EVAL SUMMARY — tickets {start} to {end - 1} (n={n})
{"=" * 50}
Inaccurate responses:  {grounding}
Off-topic responses:   {goal_drift}
Wrong outcome:         {disposition_err}
Format issues:         {schema_fails}
Total issues:          {any_fail} / {n}
Needs human review:    {needs_review}
Agent confidence avg:  {conf_mean} (note: high confidence even on failures — not a reliable signal)

TICKET RESULTS
{"=" * 50}
{chr(10).join(ticket_lines)}

FAILURE PATTERNS
{"=" * 50}
{chr(10).join(pattern_lines) if pattern_lines else "  No failure patterns identified."}

Tip: call get_ticket_detail(ticket_id=<id>) for the full breakdown of any ticket above.
"""
    return output


@mcp.tool()
def get_ticket_detail(ticket_id: int) -> str:
    """
    Get the full quality breakdown for a specific ticket from the last eval run.

    Shows the customer message, agent response, per-check results with reasons,
    and the caveat annotation explaining any ambiguity.

    Args:
        ticket_id: The ticket ID to look up (must have been in the last run_eval call)

    Returns:
        Full drill-down for the ticket.
    """
    results = _last_run.get("results", [])
    clusters = _last_run.get("clusters", [])
    caveats = _last_run.get("caveats", [])

    if not results:
        return "No eval results found. Run run_eval() first."

    match = next((r for r in results if r["eval"]["ticket_id"] == ticket_id), None)
    if not match:
        return f"Ticket #{ticket_id} not found in the last run. Available IDs: {[r['eval']['ticket_id'] for r in results]}"

    ev = match["eval"]

    # Find caveat
    caveat = None
    for i, cl in enumerate(clusters):
        if ticket_id in cl.get("tickets", []):
            if i < len(caveats):
                caveat = caveats[i]

    def check(label, failed, reason, votes=""):
        status = "FAIL" if failed else "PASS"
        vote_str = f" (votes: {votes})" if votes else ""
        return f"  {status} — {label}{vote_str}\n        {reason}"

    checks = "\n".join([
        check(
            "Factual accuracy",
            ev["hallucination"]["hallucinated"],
            ev["hallucination"].get("reason", ""),
            ev["hallucination"].get("votes", ""),
        ),
        check(
            "Stayed on topic",
            ev["goal_drift"]["drifted"],
            ev["goal_drift"].get("reason", ""),
            ev["goal_drift"].get("votes", ""),
        ),
        check(
            "Correct outcome",
            ev["wrong_disposition"]["wrong_disposition"],
            ev["wrong_disposition"].get("reason", ""),
            ev["wrong_disposition"].get("votes", ""),
        ),
    ])

    schema = ev["schema"]
    if schema["passed"]:
        schema_str = "  PASS — Format checks"
    else:
        schema_str = "\n".join(f"  FAIL — {k}: {v}" for k, v in schema["violations"].items())

    caveat_str = ""
    if caveat:
        caveat_str = f"""
AMBIGUITY NOTE
{"=" * 50}
Pattern:               {caveat.cluster_label}
Why it's grey:         {caveat.ambiguity}
Human vs AI gap:       {caveat.human_llm_gap}
Reviewer should check: {caveat.reviewer_signal}
Human review needed:   {"Yes" if caveat.needs_human_review else "No"}"""

    output = f"""TICKET #{ticket_id} — {match['disposition']} (confidence: {match['confidence']})
{"=" * 50}
CUSTOMER MESSAGE
{match['query']}

AGENT RESPONSE
{match['response']}

QUALITY CHECKS
{"=" * 50}
{checks}
{schema_str}
{caveat_str}
"""
    return output


if __name__ == "__main__":
    mcp.run(transport="stdio")
