import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Chatmaxxing Eval", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "clusters" not in st.session_state:
    st.session_state.clusters = None
if "caveats" not in st.session_state:
    st.session_state.caveats = None
if "flagged_ids" not in st.session_state:
    st.session_state.flagged_ids = set()
if "selected_ticket" not in st.session_state:
    st.session_state.selected_ticket = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def failure_tags(r: dict) -> list[str]:
    tags = []
    ev = r["eval"]
    if ev["hallucination"]["hallucinated"]:
        tags.append("Grounding")
    if ev["goal_drift"]["drifted"]:
        tags.append("Goal Drift")
    if ev["wrong_disposition"]["wrong_disposition"]:
        tags.append("Disposition")
    if not ev["schema"]["passed"]:
        tags.append("Schema")
    return tags


def ticket_row(r: dict) -> dict:
    tags = failure_tags(r)
    ev = r["eval"]
    return {
        "ticket_id": ev["ticket_id"],
        "disposition": r["disposition"],
        "confidence": r.get("confidence", "—"),
        "failures": ", ".join(tags) if tags else "✓ Clean",
        "schema_violations": ev["schema"]["violation_count"],
        "flagged": "⚑" if ev["ticket_id"] in st.session_state.flagged_ids else "",
        "_result": r,
    }


def caveat_for_ticket(ticket_id, clusters, caveats):
    """Return the caveat annotation for the cluster this ticket belongs to."""
    if not clusters or not caveats:
        return None
    for i, cl in enumerate(clusters):
        if ticket_id in cl.get("tickets", []):
            if i < len(caveats):
                return caveats[i]
    return None


def run_pipeline(start: int, end: int):
    from agent.agent import load_tickets, run_agent
    from eval.detectors import run_all_detectors
    from eval.clustering import cluster_failures
    from eval.fix_proposer import generate_caveats

    df_raw = load_tickets("data/twitter_clean.csv")
    df = df_raw[df_raw["Ticket Description"].notna()].reset_index(drop=True)
    df = df[df["Ticket Description"].str.len() > 100].reset_index(drop=True)

    results = []
    progress = st.progress(0, text="Running eval pipeline...")
    total = end - start

    for idx, i in enumerate(range(start, end)):
        row = df.iloc[i].to_dict()
        agent_out = run_agent(row, use_grounding_gate=False)
        eval_result = run_all_detectors(
            ticket_id=agent_out["ticket_id"],
            query=agent_out["query"],
            response=agent_out["response"],
            ticket_type=agent_out["ticket_type"],
            context=agent_out["context"],
            disposition=agent_out["disposition"],
            confidence=agent_out["confidence"],
        )
        results.append({
            "eval": eval_result,
            "gate_triggered": agent_out["gate_triggered"],
            "disposition": agent_out["disposition"],
            "confidence": agent_out["confidence"],
            "query": agent_out["query"],
            "response": agent_out["response"],
        })
        progress.progress((idx + 1) / total, text=f"Ticket {idx + 1}/{total}...")

    progress.empty()

    all_evals = [r["eval"] for r in results]
    clusters = cluster_failures(all_evals)
    caveats = generate_caveats(clusters) if clusters else []

    return results, clusters, caveats


# ---------------------------------------------------------------------------
# Sidebar: run controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Chatmaxxing")
    st.caption("AI support agent eval dashboard")
    st.divider()

    st.subheader("Run eval")
    col1, col2 = st.columns(2)
    with col1:
        start_idx = st.number_input("Start", min_value=0, max_value=990, value=425, step=1)
    with col2:
        end_idx = st.number_input("End", min_value=1, max_value=1000, value=445, step=1)

    if st.button("▶ Run pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running…"):
            results, clusters, caveats = run_pipeline(int(start_idx), int(end_idx))
            st.session_state.results = results
            st.session_state.clusters = clusters
            st.session_state.caveats = caveats
            st.session_state.selected_ticket = None
        st.success(f"Done — {len(results)} tickets evaluated")

    st.divider()
    if st.session_state.flagged_ids:
        st.subheader("Flagged for review")
        st.caption(f"{len(st.session_state.flagged_ids)} ticket(s)")
        st.code("\n".join(str(i) for i in sorted(st.session_state.flagged_ids)))
        if st.button("Clear flags"):
            st.session_state.flagged_ids = set()
            st.rerun()


# ---------------------------------------------------------------------------
# No data yet
# ---------------------------------------------------------------------------
if st.session_state.results is None:
    st.title("Chatmaxxing Eval")
    st.info("Configure a ticket range in the sidebar and click **Run pipeline** to start.")
    st.stop()

results = st.session_state.results
clusters = st.session_state.clusters
caveats = st.session_state.caveats
n = len(results)

# ---------------------------------------------------------------------------
# Screen 1: Metrics overview
# ---------------------------------------------------------------------------
st.title("Eval results")

grounding = sum(1 for r in results if r["eval"]["hallucination"]["hallucinated"])
goal_drift = sum(1 for r in results if r["eval"]["goal_drift"]["drifted"])
disposition = sum(1 for r in results if r["eval"]["wrong_disposition"]["wrong_disposition"])
schema_fails = sum(1 for r in results if not r["eval"]["schema"]["passed"])
any_fail = sum(1 for r in results if r["eval"]["any_failure"])
caveat_flagged = sum(
    1 for r in results
    if caveat_for_ticket(r["eval"]["ticket_id"], clusters, caveats) is not None
    and getattr(caveat_for_ticket(r["eval"]["ticket_id"], clusters, caveats), "needs_human_review", False)
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Grounding failures", grounding, delta=None)
m2.metric("Disposition errors", disposition, delta=None)
m3.metric("Schema violations", schema_fails, delta=None)
m4.metric("Needs human review", caveat_flagged, delta=None)

st.caption(f"n={n} tickets  ·  {any_fail} total failures  ·  {n - any_fail} clean")
st.divider()

# ---------------------------------------------------------------------------
# Screen 2: Ticket table with filters
# ---------------------------------------------------------------------------
st.subheader("Tickets")

all_failure_types = ["Grounding", "Goal Drift", "Disposition", "Schema"]
filter_col, _, search_col = st.columns([2, 1, 3])
with filter_col:
    filter_type = st.multiselect("Filter by failure type", all_failure_types, placeholder="All tickets")
with search_col:
    search_query = st.text_input("Search customer message", placeholder="e.g. billing, refund…")

rows = [ticket_row(r) for r in results]

if filter_type:
    rows = [r for r in rows if any(ft in r["failures"] for ft in filter_type)]
if search_query:
    sq = search_query.lower()
    rows = [r for r in rows if sq in r["_result"]["query"].lower()]

if not rows:
    st.warning("No tickets match the current filters.")
else:
    header = st.columns([1, 2, 2, 3, 1, 1])
    header[0].markdown("**ID**")
    header[1].markdown("**Disposition**")
    header[2].markdown("**Confidence**")
    header[3].markdown("**Failures**")
    header[4].markdown("**Schema**")
    header[5].markdown("**⚑**")

    for row in rows:
        tid = row["ticket_id"]
        cols = st.columns([1, 2, 2, 3, 1, 1])
        if cols[0].button(str(tid), key=f"select_{tid}", use_container_width=True):
            st.session_state.selected_ticket = tid
            st.rerun()
        cols[1].write(row["disposition"])
        cols[2].write(f"{row['confidence']:.2f}" if isinstance(row["confidence"], float) else row["confidence"])
        if row["failures"] == "✓ Clean":
            cols[3].success("✓ Clean")
        else:
            cols[3].error(row["failures"])
        cols[4].write(f"{row['schema_violations']} violation(s)" if row["schema_violations"] else "✓")
        flag_label = "⚑ unflag" if tid in st.session_state.flagged_ids else "⚑ flag"
        if cols[5].button(flag_label, key=f"flag_{tid}"):
            if tid in st.session_state.flagged_ids:
                st.session_state.flagged_ids.discard(tid)
            else:
                st.session_state.flagged_ids.add(tid)
            st.rerun()

# ---------------------------------------------------------------------------
# Screen 3: Drill-down
# ---------------------------------------------------------------------------
if st.session_state.selected_ticket is not None:
    st.divider()
    tid = st.session_state.selected_ticket
    match = next((r for r in results if r["eval"]["ticket_id"] == tid), None)

    if match:
        st.subheader(f"Ticket #{tid}")

        left, right = st.columns(2)

        with left:
            st.markdown("**Customer message**")
            st.text_area("", value=match["query"], height=180, disabled=True, key="query_area")

            st.markdown("**Agent response**")
            st.text_area("", value=match["response"], height=180, disabled=True, key="response_area")

        with right:
            st.markdown("**Detector results**")
            ev = match["eval"]

            def status(flag): return "❌" if flag else "✅"

            st.write(f"{status(ev['hallucination']['hallucinated'])} **Grounding** — {ev['hallucination'].get('reason','')}")
            st.write(f"{status(ev['goal_drift']['drifted'])} **Goal drift** — {ev['goal_drift'].get('reason','')}")
            st.write(f"{status(ev['wrong_disposition']['wrong_disposition'])} **Disposition** — {ev['wrong_disposition'].get('reason','')}")

            schema = ev["schema"]
            if schema["passed"]:
                st.write("✅ **Schema** — all checks passed")
            else:
                for check, msg in schema["violations"].items():
                    st.write(f"❌ **Schema / {check}** — {msg}")

            st.divider()
            st.markdown("**Caveat annotation**")
            caveat = caveat_for_ticket(tid, clusters, caveats)
            if caveat:
                st.markdown(f"**Pattern:** {caveat.cluster_label}")
                st.markdown(f"**Ambiguity:** {caveat.ambiguity}")
                st.markdown(f"**Human/LLM gap:** {caveat.human_llm_gap}")
                st.markdown(f"**Reviewer should:** {caveat.reviewer_signal}")
                review_badge = "🔴 Yes" if caveat.needs_human_review else "🟢 No"
                st.markdown(f"**Human review needed:** {review_badge}")
            else:
                st.caption("No caveat annotation for this ticket.")

        if st.button("⚑ Flag for review" if tid not in st.session_state.flagged_ids else "⚑ Unflag", key="drill_flag"):
            if tid in st.session_state.flagged_ids:
                st.session_state.flagged_ids.discard(tid)
            else:
                st.session_state.flagged_ids.add(tid)
            st.rerun()

        if st.button("✕ Close", key="close_drill"):
            st.session_state.selected_ticket = None
            st.rerun()
