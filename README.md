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

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd chatmaxxing
uv pip install -r requirements.txt
uv pip install mcp streamlit
```

### 2. Add your API keys

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com).  
Langfuse is optional — it records traces. Sign up at [langfuse.com](https://langfuse.com) or remove the langfuse calls if you don't need tracing.

### 3. Add your ticket data

Place your CSV at `data/twitter_clean.csv`. The pipeline expects a column called `Ticket Description`.

---

## Running the dashboard

```bash
uv run streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501), set a ticket range, and click **Run analysis**.

---

## Running as an MCP server

The MCP server exposes two tools that any MCP-compatible client (Claude Desktop, Claude Code) can call:

| Tool | What it does |
|------|-------------|
| `run_eval(start, end)` | Runs the full pipeline on a ticket range and returns a summary |
| `get_ticket_detail(ticket_id)` | Returns the full breakdown for a specific ticket from the last run |

### Connect to Claude Desktop

Add this to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "chatmaxxing": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/absolute/path/to/chatmaxxing"
    }
  }
}
```

Replace `/absolute/path/to/chatmaxxing` with the actual path on your machine.

### Connect to Claude Code

```bash
claude mcp add chatmaxxing -- uv run python /absolute/path/to/chatmaxxing/mcp_server.py
```

### Example usage in Claude

Once connected, you can say:
- *"Run eval on tickets 425 to 445"*
- *"Show me the full breakdown for ticket 6969"*
- *"Which tickets need human review?"*

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
    ├── dashboard/app.py     ← Streamlit UI
    └── mcp_server.py        ← MCP tools
```

---

## Key design decisions

**3-vote majority voting** — each detector calls Gemini 3 times and takes the majority. A 2/3 split sets `uncertain=True`, which propagates into the caveat layer. This surfaces cases where even the judge is unsure, instead of hiding the uncertainty behind a single verdict.

**Deterministic schema validator** — runs without any LLM calls. Checks escalation coherence (ESCALATE disposition must not contain resolution language), response length bounds, and disposition enum validity.

**Caveat annotations instead of auto-fixes** — the pipeline deliberately does not suggest prompt patches. It surfaces *why* a failure is ambiguous and routes it to a human. This is the right call for cases where the rubric itself is contested.
