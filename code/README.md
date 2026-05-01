# Support Triage Agent

## Setup

```bash
cd hackerrank-orchestrate-may26
pip install -r code/requirements.txt
cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=your_key_here
```

> **Free options**: Sign up at console.anthropic.com for $5 free credits
> (no card needed), or sign up at console.groq.com for a completely free
> Groq API key and store it as ANTHROPIC_API_KEY in your .env file.

## Run

```bash
python code/main.py
```

**Reads:**  `support_tickets/support_tickets.csv`
**Writes:** `support_tickets/output.csv`

## Architecture

The agent uses a **retrieval-augmented generation (RAG)** pipeline with four stages:

1. **Retrieval** (`retriever.py`): At startup, all 774 markdown files under `data/` are read, stripped of markdown syntax, and chunked into ~400-token segments with 50-token overlap. A BM25Okapi index is built over all chunks, plus per-company sub-indices for biased retrieval. For each ticket, the top-5 most relevant chunks are retrieved — biased toward the ticket's company when known.

2. **Classification** (`classifier.py`): A lightweight rule-based classifier uses keyword patterns to categorize tickets as `product_issue`, `feature_request`, `bug`, or `invalid`. Only ambiguous cases fall through to an LLM call, saving latency and API cost.

3. **Safety Check** (`safety.py`): An explicit pattern-matching guard checks for fraud, account security, legal threats, data privacy requests, assessment integrity disputes, abusive content, prompt injection, and PII. If any trigger fires, the ticket is immediately escalated. Additionally, tickets where no relevant corpus chunks are found (low BM25 score) for a known company are escalated.

4. **LLM Synthesis** (`agent.py`): Safe tickets are sent to Claude Sonnet with the retrieved chunks as grounding context. The system prompt strictly instructs the LLM to answer only from provided documents and never fabricate policies. The response is returned as structured JSON.

All prompts are centralized in `prompts.py` for easy review and version control.

## Key Design Decisions

- **BM25 over vector embeddings**: Simpler, faster, no external DB needed, and performs well on structured support docs.
- **Company-biased retrieval**: Searches the company-specific sub-index first, with fallback to the full corpus for cross-domain relevance.
- **Explicit escalation rules**: Safety-critical routing uses auditable regex patterns rather than LLM judgment.
- **Graceful error handling**: Per-row error catching ensures one bad ticket never crashes the whole run.
- **Determinism**: Temperature=0 for all LLM calls; seeded indexing.
