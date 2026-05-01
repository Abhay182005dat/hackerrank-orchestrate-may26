"""
agent.py — Core triage agent orchestration.

Pipeline per ticket:
1. Retrieve relevant corpus chunks (retriever.py)
2. Classify request type (classifier.py)
3. Check escalation safety (safety.py)
4. If safe: synthesize response via LLM with retrieved chunks as grounding
5. If unsafe: escalate with reason
6. Return TriageResult dataclass

Design decisions:
- The LLM is only called for safe tickets — escalated tickets get a standard
  response without consuming API quota.
- JSON parsing of LLM output has fallback extraction in case the model
  wraps output in code fences or adds extra text.
- Temperature=0 for determinism.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq

from classifier import classify_request_type
from prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT
from retriever import Chunk, retrieve
from safety import should_escalate


@dataclass
class TriageResult:
    """Result of triaging a single support ticket."""
    status: str            # "replied" or "escalated"
    product_area: str      # most relevant support category
    response: str          # user-facing answer
    justification: str     # concise explanation of decision
    request_type: str      # product_issue / feature_request / bug / invalid


# --------------------------------------------------------------------------- #
# LLM client (reused across calls)
# --------------------------------------------------------------------------- #
_client: Optional[Groq] = None


def _get_client() -> Groq:
    """Lazy-initialize the Groq client."""
    global _client
    if _client is None:
        api_key = os.environ.get("API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "API_KEY not set. Add it to .env or environment."
            )
        _client = Groq(api_key=api_key)
    return _client


# --------------------------------------------------------------------------- #
# Chunk formatting for the LLM prompt
# --------------------------------------------------------------------------- #
def _format_chunks(chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a readable string for the LLM context.
    Truncates each chunk to save tokens on free-tier APIs."""
    if not chunks:
        return "(No relevant documents found)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Truncate chunk text to ~300 chars to stay within rate limits
        text = chunk.text[:500]
        parts.append(
            f"[Doc {i}] (Source: {chunk.source_file}, "
            f"Company: {chunk.company})\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(parts)


def _infer_product_area(chunks: list[Chunk], issue: str,
                        company: str) -> str:
    """Infer product area from retrieved chunks' source paths."""
    if not chunks:
        return "general_support"

    # Use the top chunk's source path to determine area
    top_source = chunks[0].source_file
    parts = [p for p in top_source.replace('\\', '/').split('/') if p]

    # Skip company name and filename, use middle path components
    if len(parts) >= 3:
        # e.g., hackerrank/screen/test-settings/file.md -> screen
        area = parts[1]  # first subdirectory after company
        return area.replace('-', '_')
    elif len(parts) >= 2:
        return parts[0].replace('-', '_')

    return "general_support"


# --------------------------------------------------------------------------- #
# JSON parsing helpers
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM response, handling code fences, extra text,
    and truncated responses.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: extract fields from truncated JSON using regex
    # This handles cases where the LLM output was cut off by max_tokens
    result = {}
    for key in ['status', 'product_area', 'response', 'justification', 'request_type']:
        match = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if match:
            result[key] = match.group(1).replace('\\n', '\n').replace('\\"', '"')

    if result:
        return result

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


# --------------------------------------------------------------------------- #
# Main agent function
# --------------------------------------------------------------------------- #
def triage_ticket(issue: str, subject: str, company: str) -> TriageResult:
    """
    Triage a single support ticket through the full pipeline:
    retrieve → classify → safety check → (LLM synthesis or escalation).
    """
    # Normalize company
    company_clean = company.strip() if company else ""
    if company_clean.lower() in ('none', ''):
        company_for_retrieval = None
        company_label = "Unknown"
    else:
        company_for_retrieval = company_clean
        company_label = company_clean

    # Step 1: Retrieve relevant corpus chunks
    query = f"{subject} {issue}".strip()
    chunks = retrieve(query, company=company_for_retrieval, top_k=3)

    # Step 2: Classify request type
    request_type = classify_request_type(issue, subject, company_label)

    # Step 3: Safety / escalation check
    escalate, escalation_reason = should_escalate(
        issue, chunks, subject=subject, company=company_for_retrieval
    )

    # Infer product area from retrieval
    product_area = _infer_product_area(chunks, issue, company_label)

    # Step 4: Handle escalation vs. safe reply
    if escalate:
        # For adversarial / injection content, mark as invalid
        if "adversarial" in escalation_reason.lower() or "injection" in escalation_reason.lower():
            request_type = "invalid"

        return TriageResult(
            status="escalated",
            product_area=product_area,
            response=(
                "This issue requires specialized attention from our support team. "
                "A support specialist will review your case and follow up with you shortly. "
                "We appreciate your patience."
            ),
            justification=escalation_reason,
            request_type=request_type,
        )

    # Step 5: LLM synthesis for safe tickets
    try:
        result = _llm_synthesize(issue, subject, company_label, chunks)

        # Validate and override with our own classification if needed
        # (LLM might disagree, but our pipeline's classification is more consistent)
        result.request_type = request_type

        # Ensure product_area from LLM is used if it's more specific
        if not result.product_area or result.product_area.strip() == "":
            result.product_area = product_area

        return result

    except Exception as e:
        # If LLM call fails, return a graceful escalation
        print(f"[agent] LLM synthesis failed: {e}")
        return TriageResult(
            status="escalated",
            product_area=product_area,
            response=(
                "I was unable to process this request at this time. "
                "A support specialist will follow up with you."
            ),
            justification=f"LLM synthesis failed: {str(e)[:100]}",
            request_type=request_type,
        )


def _llm_synthesize(issue: str, subject: str, company: str,
                    chunks: list[Chunk],
                    max_retries: int = 3) -> TriageResult:
    """Call the LLM with retrieved chunks to produce a grounded response.
    Retries with exponential backoff on connection errors."""
    client = _get_client()

    user_prompt = TRIAGE_USER_PROMPT.format(
        company=company,
        subject=subject,
        issue=issue,
        chunks=_format_chunks(chunks),
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=768,
                temperature=0,
                messages=[
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_text = response.choices[0].message.content
            parsed = _extract_json(raw_text)

            # Validate required fields
            status = parsed.get("status", "replied")
            if status not in ("replied", "escalated"):
                status = "escalated"  # default to safe option

            return TriageResult(
                status=status,
                product_area=parsed.get("product_area", "general_support"),
                response=parsed.get("response", ""),
                justification=parsed.get("justification", ""),
                request_type=parsed.get("request_type", "product_issue"),
            )
        except Exception as e:
            last_error = e
            wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
            print(f"[agent] LLM attempt {attempt+1}/{max_retries} failed: {e}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    raise last_error  # re-raise after all retries exhausted
