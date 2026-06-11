import sys
sys.path.append(".")

import json
import os
import streamlit as st
import pandas as pd

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cached_results.json")


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return None, None, None
    with open(CACHE_PATH) as f:
        data = json.load(f)
    from eval.fix_proposer import CaveatAnnotation
    results = data["results"]
    clusters = data["clusters"]
    caveats = [CaveatAnnotation(**c) for c in data["caveats"]]
    return results, clusters, caveats


def save_cache(results, clusters, caveats):
    with open(CACHE_PATH, "w") as f:
        json.dump({
            "results": results,
            "clusters": clusters,
            "caveats": [c.model_dump() for c in caveats],
        }, f, indent=2)

st.set_page_config(page_title="AI Agent Quality & Performance Hub", layout="wide")

# ---------------------------------------------------------------------------
# Page 1 — landing: browse cached demo data, or enter a key to run live
# ---------------------------------------------------------------------------
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = None
if "entered_app" not in st.session_state:
    st.session_state.entered_app = False

if not st.session_state.entered_app:
    st.title("AI Agent Quality & Performance Hub")
    st.markdown(
        "This dashboard evaluates AI customer-support agent responses for "
        "factual accuracy, goal drift, and wrong outcomes."
    )
    st.markdown("**Browse the pre-computed demo run** — no key needed, loads instantly:")
    if st.button("View demo dashboard", type="primary"):
        st.session_state.entered_app = True
        st.rerun()

    st.divider()
    st.markdown(
        "**Run live analysis on fresh tickets** — enter a Gemini API key. "
        "The key is used only for this session and is never stored."
    )
    st.info(
        "**Note on the free tier:** Google's free Gemini API allows 10 requests "
        "per minute. Each ticket this app evaluates uses ~10 requests (one agent "
        "reply plus three detectors voting three times each), so a live run takes "
        "roughly **one minute per ticket** and large ranges will be slow. "
        "If you just want to see the full output, click **View demo dashboard** "
        "above — it loads a complete pre-computed run instantly, no key required.",
        icon="⏱️",
    )
    key_input = st.text_input("Gemini API key (optional)", type="password", placeholder="AIza…")
    if st.button("Enter key and continue", disabled=not key_input):
        st.session_state.gemini_api_key = key_input
        st.session_state.entered_app = True
        st.rerun()
    st.stop()

# Build the client only if a key was provided. Browsing cached data needs no client.
_gemini_client = None
if st.session_state.gemini_api_key:
    from google import genai as _genai
    _gemini_client = _genai.Client(api_key=st.session_state.gemini_api_key)

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


def caveat_for_ticket(ticket_id, clusters, caveats):
    if not clusters or not caveats:
        return None
    for i, cl in enumerate(clusters):
        if ticket_id in cl.get("tickets", []):
            if i < len(caveats):
                return caveats[i]
    return None


def run_pipeline(start: int, end: int, gemini_client=None):
    from agent.agent import load_tickets, run_agent
    from eval.detectors import run_all_detectors
    from eval.clustering import cluster_failures
    from eval.fix_proposer import generate_caveats

    df_raw = load_tickets("data/twitter_clean.csv")
    df = df_raw[df_raw["Ticket Description"].notna()].reset_index(drop=True)
    df = df[df["Ticket Description"].str.len() > 100].reset_index(drop=True)

    results = []
    progress = st.progress(0, text="Analysing tickets…")
    total = end - start

    for idx, i in enumerate(range(start, end)):
        row = df.iloc[i].to_dict()
        agent_out = run_agent(row, use_grounding_gate=False, gemini_client=gemini_client)
        eval_result = run_all_detectors(
            ticket_id=agent_out["ticket_id"],
            query=agent_out["query"],
            response=agent_out["response"],
            ticket_type=agent_out["ticket_type"],
            context=agent_out["context"],
            disposition=agent_out["disposition"],
            gemini_client=gemini_client,
        )
        results.append({
            "eval": eval_result,
            "gate_triggered": agent_out["gate_triggered"],
            "disposition": agent_out["disposition"],
            "confidence": agent_out["confidence"],
            "query": agent_out["query"],
            "response": agent_out["response"],
        })
        progress.progress((idx + 1) / total, text=f"Ticket {idx + 1} of {total}…")

    progress.empty()

    all_evals = [r["eval"] for r in results]
    clusters = cluster_failures(all_evals, gemini_client=gemini_client)
    caveats = generate_caveats(clusters, gemini_client=gemini_client) if clusters else []

    return results, clusters, caveats


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Chatmaxxing")
    st.caption("AI Agent Quality & Performance Hub")
    st.divider()

    st.subheader("Select ticket range")
    ticket_range = st.slider(
        "Ticket range",
        min_value=0,
        max_value=999,
        value=(425, 427),
        label_visibility="collapsed",
    )
    col1, col2 = st.columns(2)
    with col1:
        start_idx = st.number_input("From", min_value=0, max_value=999, value=ticket_range[0], step=1)
    with col2:
        end_idx = st.number_input("To", min_value=1, max_value=1000, value=ticket_range[1], step=1)
    st.caption(f"{end_idx - start_idx} tickets selected")
    st.caption(
        "ℹ️ Each ticket uses ~10 API calls. On the Gemini free tier (10 calls/min) "
        "expect roughly 1 minute per ticket. The dashboard below already shows a "
        "full pre-computed run — you only need to run fresh tickets to see live analysis."
    )

    if _gemini_client is None:
        st.info("Viewing demo data. To run live analysis, reload and enter a Gemini API key.")
    run_clicked = st.button(
        "▶ Run analysis",
        use_container_width=True,
        type="primary",
        disabled=_gemini_client is None,
    )
    if run_clicked:
        with st.spinner("Running…"):
            results, clusters, caveats = run_pipeline(start_idx, end_idx, gemini_client=_gemini_client)
            save_cache(results, clusters, caveats)
            st.session_state.results = results
            st.session_state.clusters = clusters
            st.session_state.caveats = caveats
            st.session_state.selected_ticket = None
        st.success(f"Done — {len(results)} tickets analysed")

    st.divider()
    if st.session_state.flagged_ids:
        st.subheader("Flagged for review")
        st.caption(f"{len(st.session_state.flagged_ids)} ticket(s)")
        st.code("\n".join(str(i) for i in sorted(st.session_state.flagged_ids)))
        if st.button("Clear flags"):
            st.session_state.flagged_ids = set()
            st.rerun()


# ---------------------------------------------------------------------------
# No data yet — try loading from cache
# ---------------------------------------------------------------------------
if st.session_state.results is None:
    cached = load_cache()
    if cached[0] is not None:
        st.session_state.results, st.session_state.clusters, st.session_state.caveats = cached

if st.session_state.results is None:
    st.title("AI Agent Quality & Performance Hub")
    st.info("Select a ticket range in the sidebar and click **Run analysis** to get started.")
    st.stop()

results = st.session_state.results
clusters = st.session_state.clusters
caveats = st.session_state.caveats
n = len(results)

# ---------------------------------------------------------------------------
# Metrics overview
# ---------------------------------------------------------------------------
st.title("AI Agent Quality & Performance Hub")

if _gemini_client is None:
    st.caption(
        "📊 Showing a pre-computed demo run. To evaluate fresh tickets live, "
        "reload and enter a Gemini API key (note: free tier is ~1 min per ticket)."
    )

grounding = sum(1 for r in results if r["eval"]["hallucination"]["hallucinated"])
disposition_err = sum(1 for r in results if r["eval"]["wrong_disposition"]["wrong_disposition"])
schema_fails = sum(1 for r in results if not r["eval"]["schema"]["passed"])
any_fail = sum(1 for r in results if r["eval"]["any_failure"])

caveat_map = {}
for r in results:
    tid = r["eval"]["ticket_id"]
    c = caveat_for_ticket(tid, clusters, caveats)
    if c is not None:
        caveat_map[tid] = c

needs_review = sum(1 for c in caveat_map.values() if getattr(c, "needs_human_review", False))

confidences = [r["confidence"] for r in results if isinstance(r.get("confidence"), float)]
conf_mean = round(sum(confidences) / len(confidences), 2) if confidences else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Inaccurate responses", grounding, help="Agent responses that introduced facts not present in the conversation")
m2.metric("Incorrect outcomes", disposition_err, help="Cases where the agent chose the wrong resolution type")
m3.metric("Format issues", schema_fails, help="Responses that violated structural rules")
m4.metric("Needs human review", needs_review, help="Cases where automated checks were inconclusive")

st.caption(f"{n} tickets analysed  ·  {any_fail} issues found  ·  {n - any_fail} clean")
if conf_mean is not None:
    st.caption(
        f"⚠ Note: the agent rated itself {conf_mean:.0%} confident on average — including on tickets where it made errors. "
        "Confidence scores alone are not a reliable quality signal; use the issue flags and review markers below instead."
    )
st.divider()

# ---------------------------------------------------------------------------
# Ticket table
# ---------------------------------------------------------------------------
st.subheader("Ticket breakdown")

all_failure_types = ["Grounding", "Goal Drift", "Disposition", "Schema"]
filter_col, _, search_col = st.columns([2, 1, 3])
with filter_col:
    filter_type = st.multiselect("Filter by issue type", all_failure_types, placeholder="Show all tickets")
with search_col:
    search_query = st.text_input("Search by customer message", placeholder="e.g. billing, refund, account…")

# Build filtered list
filtered = results
if filter_type:
    filtered = [r for r in filtered if any(ft in failure_tags(r) for ft in filter_type)]
if search_query:
    sq = search_query.lower()
    filtered = [r for r in filtered if sq in r["query"].lower()]

if not filtered:
    st.warning("No tickets match the current filters.")
else:
    # Build display dataframe
    rows = []
    for r in filtered:
        tags = failure_tags(r)
        ev = r["eval"]
        tid = ev["ticket_id"]
        if tags:
            status_md = "✗ " + ", ".join(tags)
        else:
            status_md = "✓ Clean"
        rows.append({
            "Ticket ID": str(tid),
            "Outcome": r["disposition"],
            "Confidence": f"{r['confidence']:.0%}" if isinstance(r.get("confidence"), float) else "—",
            "Issues Found": status_md,
            "Flagged": "⚑" if tid in st.session_state.flagged_ids else "",
        })

    df_display = pd.DataFrame(rows)
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticket ID": st.column_config.TextColumn(width="small"),
            "Outcome": st.column_config.TextColumn(width="small"),
            "Confidence": st.column_config.TextColumn(width="small"),
            "Issues Found": st.column_config.TextColumn(width="large"),
            "Flagged": st.column_config.TextColumn(width="small"),
        },
    )

    st.caption("Click a ticket below to inspect it in detail.")

    # Inline expanders — one per ticket, opens in place
    for r in filtered:
        ev = r["eval"]
        tid = ev["ticket_id"]
        tags = failure_tags(r)
        label = f"{'✗' if tags else '✓'} Ticket #{tid} — {r['disposition']}"
        if tags:
            label += f"  ·  {', '.join(tags)}"

        with st.expander(label):
            left, right = st.columns(2)

            with left:
                st.markdown("**What the customer said**")
                st.text_area(
                    "Customer message",
                    value=r["query"],
                    height=160,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"query_{tid}",
                )
                st.markdown("**What the agent replied**")
                st.text_area(
                    "Agent response",
                    value=r["response"],
                    height=160,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"response_{tid}",
                )

            with right:
                st.markdown("**Quality checks**")

                def _row(label, failed, reason):
                    icon = "✗" if failed else "✓"
                    st.markdown(f"{icon} **{label}** — {reason}")

                _row(
                    "Factual accuracy",
                    ev["hallucination"]["hallucinated"],
                    ev["hallucination"].get("reason", ""),
                )
                _row(
                    "Stayed on topic",
                    ev["goal_drift"]["drifted"],
                    ev["goal_drift"].get("reason", ""),
                )
                _row(
                    "Correct outcome",
                    ev["wrong_disposition"]["wrong_disposition"],
                    ev["wrong_disposition"].get("reason", ""),
                )

                schema = ev["schema"]
                if schema["passed"]:
                    st.markdown("✓ **Format** — all checks passed")
                else:
                    for check, msg in schema["violations"].items():
                        st.markdown(f"✗ **Format / {check}** — {msg}")

                caveat = caveat_map.get(tid)
                if caveat:
                    st.divider()
                    st.markdown("**Why this case is ambiguous**")
                    st.markdown(f"**Pattern:** {caveat.cluster_label}")
                    st.markdown(f"**What makes it grey:** {caveat.ambiguity}")
                    st.markdown(f"**Where humans and AI disagree:** {caveat.human_llm_gap}")
                    st.markdown(f"**What a reviewer should check:** {caveat.reviewer_signal}")
                    review_badge = "✓ Yes — needed" if caveat.needs_human_review else "✗ Not needed"
                    st.markdown(f"**Human review needed:** {review_badge}")

            flag_label = "⚑ Remove review flag" if tid in st.session_state.flagged_ids else "⚑ Flag for human review"
            if st.button(flag_label, key=f"flag_{tid}"):
                if tid in st.session_state.flagged_ids:
                    st.session_state.flagged_ids.discard(tid)
                else:
                    st.session_state.flagged_ids.add(tid)
                st.rerun()
