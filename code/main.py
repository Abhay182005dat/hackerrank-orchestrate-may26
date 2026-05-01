"""
main.py — CLI entry point for the support triage agent.

Usage:
    python code/main.py

Reads:  support_tickets/support_tickets.csv
Writes: support_tickets/output.csv

Design decisions:
- Resume support: if output.csv already has rows, skip those and continue
  from where we left off. This handles interruptions gracefully.
- Errors per row are caught and handled gracefully — one bad ticket never
  crashes the entire run. Failed rows get status=escalated.
- Uses tqdm for a progress indicator during processing.
- CSV output uses QUOTE_ALL for response/justification fields to safely
  handle embedded commas and newlines.
- Loads .env file for ANTHROPIC_API_KEY via python-dotenv.
"""

import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Add code/ directory to path so imports work when run from repo root
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from agent import TriageResult, triage_ticket  # noqa: E402
from retriever import init_retriever  # noqa: E402

OUTPUT_HEADER = ['status', 'product_area', 'response',
                 'justification', 'request_type']


def _load_existing_results(output_csv: Path) -> list[list[str]]:
    """Load already-processed rows from output.csv for resume support."""
    if not output_csv.exists():
        return []
    existing = []
    with open(output_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return []
        for row in reader:
            if len(row) >= 5:
                existing.append(row)
    return existing


def main():
    # Load .env from repo root
    repo_root = CODE_DIR.parent
    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        print(f"[main] Loaded .env from {dotenv_path}")
    else:
        print("[main] No .env file found — using existing environment variables")

    # Verify API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    # Paths
    input_csv = repo_root / "support_tickets" / "support_tickets.csv"
    output_csv = repo_root / "support_tickets" / "output.csv"
    data_dir = repo_root / "data"

    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}")
        sys.exit(1)

    # Step 1: Initialize the retriever (loads and indexes corpus)
    print("[main] Loading and indexing corpus...")
    init_retriever(str(data_dir))

    # Step 2: Read input tickets
    print(f"[main] Reading tickets from {input_csv}")
    tickets = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(row)

    # Step 3: Check for resume
    existing_rows = _load_existing_results(output_csv)
    start_idx = len(existing_rows)

    if start_idx > 0:
        print(f"[main] Resuming from ticket {start_idx + 1}/{len(tickets)} "
              f"({start_idx} already processed)")
    else:
        print(f"[main] Processing {len(tickets)} tickets from scratch...")

    # Step 4: Process remaining tickets
    new_results: list[TriageResult] = []
    remaining = tickets[start_idx:]
    for i, ticket in enumerate(tqdm(remaining, desc="Triaging tickets",
                                     initial=start_idx, total=len(tickets))):
        issue = ticket.get('Issue', '').strip()
        subject = ticket.get('Subject', '').strip()
        company = ticket.get('Company', '').strip()

        try:
            result = triage_ticket(issue, subject, company)
            new_results.append(result)
        except Exception as e:
            # Graceful error handling — never crash on a single ticket
            print(f"\n[main] Error on ticket {start_idx + i + 1}: {e}")
            new_results.append(TriageResult(
                status="escalated",
                product_area="general_support",
                response=(
                    "We encountered an error processing your request. "
                    "A support specialist will follow up with you."
                ),
                justification=f"Processing error: {str(e)[:100]}",
                request_type="product_issue",
            ))

    # Step 5: Write full output CSV (existing + new)
    print(f"\n[main] Writing results to {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(OUTPUT_HEADER)

        # Write previously processed rows
        for row in existing_rows:
            writer.writerow(row)

        # Write newly processed rows
        for result in new_results:
            writer.writerow([
                result.status,
                result.product_area,
                result.response,
                result.justification,
                result.request_type,
            ])

    total = len(existing_rows) + len(new_results)
    print(f"[main] Done! {total} tickets in output ({start_idx} resumed + "
          f"{len(new_results)} new).")
    print(f"[main] Output: {output_csv}")

    # Summary stats (count from all rows)
    all_statuses = [r[0] for r in existing_rows] + \
                   [r.status for r in new_results]
    replied = sum(1 for s in all_statuses if s == 'replied')
    escalated = sum(1 for s in all_statuses if s == 'escalated')
    print(f"[main] Summary: {replied} replied, {escalated} escalated")


if __name__ == "__main__":
    main()
