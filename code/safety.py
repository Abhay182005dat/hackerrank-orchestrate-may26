"""
safety.py — Escalation guard for high-risk support tickets.

Design decisions:
- Pattern-based detection is explicit and auditable vs. relying on LLM judgment
  for safety-critical routing decisions.
- Multiple escalation categories ensure broad coverage of risky scenarios.
- PII detection uses conservative regex patterns — false positives are
  acceptable (escalation is safe), false negatives are not.
- The BM25 score threshold check catches out-of-scope issues where we have
  no corpus coverage.
"""

import re
from typing import Optional

from retriever import Chunk, get_min_score_threshold

# --------------------------------------------------------------------------- #
# Escalation pattern categories
# --------------------------------------------------------------------------- #

# Fraud & financial
FRAUD_PATTERNS = [
    r'\b(fraud|stolen card|unauthorized (transaction|charge|payment)|'
    r'chargeback|charge back|dispute.{0,20}charge|'
    r'stolen.{0,15}(card|account|money)|'
    r'suspicious.{0,15}(transaction|activity|charge)|'
    r'fraudulent|scam|phishing)\b',
]

# Account security
ACCOUNT_SECURITY_PATTERNS = [
    r'\b(account.{0,10}hack|hack.{0,10}account|'
    r'identity.{0,5}(theft|stolen)|'
    r'someone.{0,20}(accessed|logged into|using).{0,10}(my|our) account|'
    r'unauthorized.{0,10}access|account.{0,10}compromised|'
    r'account.{0,10}takeover|password.{0,10}stolen)\b',
]

# Legal threats
LEGAL_PATTERNS = [
    r'\b(legal action|lawsuit|lawyer|attorney|court order|subpoena|'
    r'sue you|litigation|legal proceedings|cease and desist|'
    r'regulatory complaint|file.{0,10}complaint|'
    r'report.{0,10}(to|with).{0,10}(authorities|police|regulator))\b',
]

# Data privacy / GDPR
PRIVACY_PATTERNS = [
    r'\b(gdpr|data deletion|delete.{0,10}(my|all).{0,10}data|'
    r'right to be forgotten|data subject (access )?request|'
    r'dsar|privacy (violation|breach)|personal data.{0,10}(exposed|leaked)|'
    r'data breach)\b',
]

# Assessment integrity (HackerRank-specific)
ASSESSMENT_INTEGRITY_PATTERNS = [
    r'\b(cheat|cheating|plagiarism|plagiarized|proctoring.{0,10}(flag|violation)|'
    r'test.{0,10}(invalidat|disqualif)|result.{0,10}invalidat|'
    r'unfair.{0,10}(grading|score|evaluation)|'
    r'review my (answers|score|test)|increase my score|'
    r'graded.{0,10}unfairly|platform.{0,10}must have.{0,10}graded|'
    r'score.{0,10}dispute|wrong score)\b',
]

# Physical threats / NSFW / abuse
ABUSE_PATTERNS = [
    r'\b(kill|murder|bomb|threat|shoot|attack|assault|'
    r'physical.{0,5}(threat|harm|violence)|death threat|'
    r'hurt (you|myself|someone))\b',
]

# Prompt injection / adversarial
INJECTION_PATTERNS = [
    r'\b(ignore previous instructions|jailbreak|system prompt|'
    r'DAN|ignore all instructions|pretend you are|'
    r'act as if you|reveal your prompt|bypass|'
    r'ignore.{0,10}(rules|constraints|guidelines)|'
    r'override.{0,10}(instructions|safety)|'
    r'delete all files|rm -rf|format c:)\b',
]

# PII patterns — conservative regex
PII_PATTERNS = [
    # Credit card numbers (13-19 digits, optionally separated by spaces/dashes)
    r'\b(?:\d[ -]*?){13,19}\b',
    # SSN (US)
    r'\b\d{3}[-. ]\d{2}[-. ]\d{4}\b',
    # Passport numbers (common formats)
    r'\b[A-Z]{1,2}\d{6,9}\b',
]


# --------------------------------------------------------------------------- #
# Main escalation check
# --------------------------------------------------------------------------- #
def should_escalate(issue: str,
                    retrieved_chunks: list[Chunk],
                    subject: str = "",
                    company: Optional[str] = None) -> tuple[bool, str]:
    """
    Check whether a ticket should be escalated to a human.

    Returns:
        (True, reason) if escalation is needed.
        (False, "") if safe to handle automatically.
    """
    combined_text = f"{subject} {issue}".strip()

    # 1. Check each escalation category
    checks = [
        (FRAUD_PATTERNS, "Ticket involves potential fraud or financial dispute — requires human review."),
        (ACCOUNT_SECURITY_PATTERNS, "Ticket involves account security/takeover concerns — requires human review."),
        (LEGAL_PATTERNS, "Ticket contains legal threats or references — requires human review."),
        (PRIVACY_PATTERNS, "Ticket involves data privacy/GDPR request — requires specialized handling."),
        (ASSESSMENT_INTEGRITY_PATTERNS, "Ticket involves assessment integrity or score dispute — requires human review."),
        (ABUSE_PATTERNS, "Ticket contains potentially threatening or abusive content — requires human review."),
        (INJECTION_PATTERNS, "Ticket contains adversarial or prompt injection content — flagged as invalid."),
    ]

    for patterns, reason in checks:
        if _match_any(combined_text, patterns):
            return True, reason

    # 2. Check for PII
    for pattern in PII_PATTERNS:
        if re.search(pattern, combined_text):
            return True, "Ticket contains potential PII (credit card, SSN, passport) — requires human review to redact."

    # 3. Check corpus coverage — if no relevant chunks found for a known company
    if retrieved_chunks:
        top_score = max(c.score for c in retrieved_chunks)
        threshold = get_min_score_threshold()

        company_lower = (company or "").lower()
        if company_lower not in ('', 'none') and top_score < threshold:
            return True, (
                f"No relevant support documentation found for this {company} issue "
                f"(top BM25 score: {top_score:.2f} < threshold {threshold:.2f}) — "
                f"escalating to human specialist."
            )
    elif company and company.lower() not in ('', 'none'):
        # No chunks at all
        return True, "No relevant support documentation found — escalating to human specialist."

    return False, ""


def _match_any(text: str, patterns: list[str]) -> bool:
    """Check if any pattern matches the text."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
