"""
Turn the raw Twitter customer support CSV into conversation-level rows
that plug into the existing eval pipeline.

Input:  data/twcs.csv          (3M tweets, one per row)
Output: data/twitter_clean.csv  (multi-turn conversations, one per row)

The CURRENT MESSAGE is always the last *customer* turn, so the agent is
always responding to a customer — never echoing another agent message.
"""

import pandas as pd
from collections import defaultdict

INPUT_PATH = "data/twcs.csv"
OUTPUT_PATH = "data/twitter_clean.csv"
MIN_TURNS = 4              # only keep conversations with >= 4 turns
MAX_CONVERSATIONS = 1000   # cap so the file stays manageable

print(f"Loading {INPUT_PATH} ...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df):,} tweets")

tweets_by_id = {row["tweet_id"]: row for _, row in df.iterrows()}

# Conversation roots: customer tweets (inbound=True) that are NOT replies
roots = df[(df["inbound"] == True) & (df["in_response_to_tweet_id"].isna())]
print(f"Found {len(roots):,} conversation roots")

# Reverse index: tweet_id -> list of reply tweet_ids
replies_to = defaultdict(list)
for _, row in df.iterrows():
    parent = row["in_response_to_tweet_id"]
    if pd.notna(parent):
        replies_to[parent].append(row["tweet_id"])


def walk_conversation(root_id):
    """Follow the reply chain from a root tweet. Returns list of tweets in order."""
    chain = []
    current_id = root_id
    visited = set()
    while current_id is not None and current_id not in visited:
        visited.add(current_id)
        if current_id not in tweets_by_id:
            break
        tweet = tweets_by_id[current_id]
        chain.append(tweet)
        children = replies_to.get(current_id, [])
        current_id = children[0] if children else None
    return chain


print("Building conversations ...")
conversations = []
skipped_no_customer_last = 0

for _, root in roots.iterrows():
    chain = walk_conversation(root["tweet_id"])
    if len(chain) < MIN_TURNS:
        continue

    # --- FIX: find the index of the LAST customer (inbound) turn ---
    last_customer_idx = None
    for idx in range(len(chain) - 1, -1, -1):
        if chain[idx]["inbound"] == True:
            last_customer_idx = idx
            break

    # If there is no customer turn after the first, or the last customer turn
    # IS the root (no agent has replied yet), this isn't a usable conversation.
    if last_customer_idx is None or last_customer_idx == 0:
        skipped_no_customer_last += 1
        continue

    # History = everything BEFORE the last customer message
    history_chain = chain[:last_customer_idx]
    # Current message = the last customer message (what the agent must answer)
    current_question = chain[last_customer_idx]["text"]

    customer_turns = [t["text"] for t in history_chain if t["inbound"] == True]
    brand_turns = [t["text"] for t in history_chain if t["inbound"] == False]

    # Need at least one prior agent reply so there is real context
    if len(brand_turns) < 1:
        continue

    history_lines = []
    for t in history_chain:
        speaker = "Customer" if t["inbound"] else "Agent"
        history_lines.append(f"{speaker}: {t['text']}")
    history = "\n".join(history_lines)

    brand_tweet = next((t for t in history_chain if t["inbound"] == False), None)
    brand_name = brand_tweet["author_id"] if brand_tweet is not None else "Unknown"

    conversations.append({
        "Ticket ID": int(root["tweet_id"]),
        "Customer Name": "Customer",
        "Product Purchased": brand_name,
        "Ticket Type": "Support inquiry",
        "Ticket Subject": current_question[:60],
        "Ticket Description": f"CONVERSATION HISTORY:\n{history}\n\nCURRENT MESSAGE:\n{current_question}",
        "Customer Satisfaction Rating": None,
    })

    if len(conversations) >= MAX_CONVERSATIONS:
        break

print(f"Built {len(conversations):,} conversations with {MIN_TURNS}+ turns")
print(f"Skipped {skipped_no_customer_last:,} chains where last turn wasn't a usable customer message")

out = pd.DataFrame(conversations)
out.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
print(f"\nSample conversation:")
print(out.iloc[0]["Ticket Description"][:600])