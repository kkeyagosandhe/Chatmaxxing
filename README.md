# Chatmaxxing — AI Agent Quality & Performance Hub

An eval pipeline for AI customer support agents. Detects factual errors, off-topic responses, wrong resolution types, and format violations. Surfaces ambiguous cases for human review using majority-voted LLM judges and caveat annotations.

---

## What it does

- Runs your support agent on real tickets and scores every response
- Uses 3-vote majority voting to flag uncertain verdicts (not just pass/fail)
- Clusters failures into patterns and explains why each one is hard to judge automatically
- Surfaces a confidence miscalibration finding: the agent rates itself highly even when it fails
- Provides a Streamlit dashboard and an MCP server so it works as both a visual tool and a developer integration

---

## Before you start — what you need

- **Python 3.11 or newer** — check by running `python --version` in your terminal. If you don't have it, download it from [python.org](https://python.org).
- **uv** — a fast Python package manager. Install it by running:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **A Gemini API key** — you have two options:
  - **Option 1 (simpler):** Get a free key at [aistudio.google.com](https://aistudio.google.com). Click "Get API key", create one, and copy it. Add it to `.env` as `GEMINI_API_KEY=your_key`.
  - **Option 2 (if you have GCP credits):** Use Google Cloud Vertex AI. You will need a GCP project with Vertex AI enabled, and to run `gcloud auth application-default login` on your machine. Add `GOOGLE_CLOUD_PROJECT=your_project_id` to `.env` and update the client initialization in each file to `genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")`.
- **A Langfuse account (optional)** — for tracing. Free at [langfuse.com](https://langfuse.com). If you skip this, the pipeline still works but won't record traces.

---

## Step-by-step setup

### Step 1 — Clone the repo

Open your terminal and run:

```bash
git clone https://github.com/kkeyagosandhe/Chatmaxxing.git
cd Chatmaxxing
```

### Step 2 — Install dependencies

```bash
uv pip install -r requirements.txt
uv pip install mcp streamlit
```

This installs everything the project needs. It may take a minute.

### Step 3 — Create your `.env` file

In the `Chatmaxxing` folder, create a new file called `.env` (note the dot at the start). Open it in any text editor and paste this in:

```
GEMINI_API_KEY=paste_your_gemini_key_here
LANGFUSE_SECRET_KEY=paste_your_langfuse_secret_key_here
LANGFUSE_PUBLIC_KEY=paste_your_langfuse_public_key_here
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Replace the placeholder values with your actual keys. If you are skipping Langfuse, you can leave those three lines out entirely.

### Step 4 — Add your ticket data

Place your support ticket CSV file at `data/twitter_clean.csv`. The file must have a column called `Ticket Description`. If you are using a different CSV, make sure that column name matches.

### Step 5 — You are ready

Pick one of the two ways to use this tool below.

---

## Option A — Visual dashboard (Streamlit)

Run this command:

```bash
uv run streamlit run dashboard/app.py
```

Your browser will open automatically at [http://localhost:8501](http://localhost:8501).

**How to use it:**
1. Use the slider or the "From" / "To" boxes in the left sidebar to pick a ticket range (start small, e.g. 425 to 430, to test it out)
2. Click **Run analysis**
3. Wait for the pipeline to finish — it makes several AI calls per ticket so it takes a moment
4. The results appear on screen: a summary of issues at the top, and a list of tickets below
5. Click on any ticket row to expand it and see the full breakdown

---

## Option B — MCP server (for use inside Claude Desktop or Cursor)

This lets you run the eval pipeline directly from a chat window — just describe what you want in plain English.

### Connect to Claude Desktop

1. Open this file on your computer (create it if it does not exist):
   - **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

2. Paste this into the file, replacing the path with the actual location of your Chatmaxxing folder:

```json
{
  "mcpServers": {
    "chatmaxxing": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/absolute/path/to/Chatmaxxing"
    }
  }
}
```

3. Restart Claude Desktop. You should now see Chatmaxxing listed as a connected tool.

### Connect to Cursor

1. Open Cursor settings → MCP
2. Add a new server with:
   - **Name:** `chatmaxxing`
   - **Command:** `uv run python /absolute/path/to/Chatmaxxing/mcp_server.py`

### Connect to Claude Code (terminal)

```bash
claude mcp add chatmaxxing -- uv run python /absolute/path/to/Chatmaxxing/mcp_server.py
```

### What you can say once connected

- *"Run eval on tickets 425 to 445"*
- *"Show me the full breakdown for ticket 6969"*
- *"Which tickets need human review?"*

The MCP server uses your own `GEMINI_API_KEY` from the `.env` file. You are never charged for anyone else's usage.

---

## Architecture

```
ticket CSV
    │
    ▼
agent/agent.py          ← support agent (Gemini structured output)
    │
    ▼
eval/detectors.py       ← 3-vote LLM judges + deterministic schema validator
    │
    ▼
eval/clustering.py      ← TF-IDF + KMeans failure clustering
    │
    ▼
eval/fix_proposer.py    ← caveat annotations (ambiguity, human/LLM gap, review signal)
    │
    ├── dashboard/app.py     ← Streamlit dashboard
    └── mcp_server.py        ← MCP tools
```

---

## Eval results

**Dataset:** Twitter multi-turn customer support conversations (1,000 grouped threads, 4+ turns each)

| Metric | Result |
|--------|--------|
| Tickets evaluated | 20 |
| Any failure | 12 / 20 (60%) |
| Inaccurate responses | 1 / 20 |
| Wrong disposition | 7 / 20 (35%) |
| Needs human review | varies by run |
| Agent confidence avg | ~0.91 on failures and clean tickets alike |

**Key finding:** Disposition errors dominate at ~35%. Agent confidence is miscalibrated — it reports high confidence (~0.91) regardless of whether the response was correct or not, making confidence an unreliable quality signal on its own.

**Grounding gate impact:** Running with the grounding gate on reduced grounding failures from 1 → 0 and total failures from 8 → 7 across 20 tickets, converting ungrounded confident-but-wrong responses into safe escalations.

---

## Key design decisions

**3-vote majority voting** — each detector calls Gemini 3 times and takes the majority. A 2/3 split sets `uncertain=True`, which propagates into the caveat layer. This surfaces cases where even the judge is unsure, instead of hiding the uncertainty behind a single verdict.

**Deterministic schema validator** — runs without any LLM calls. Checks escalation coherence (an ESCALATE response must not assert the issue is resolved), response length bounds, and disposition enum validity.

**Caveat annotations instead of auto-fixes** — the pipeline deliberately does not suggest prompt patches. It surfaces why a failure is ambiguous and routes it to a human. This is the right call for cases where the rubric itself is contested.
