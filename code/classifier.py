"""
classifier.py — Lightweight rule-based + LLM-backed request type classifier.

Design decisions:
- Keyword heuristics handle clear-cut cases (bugs, feature requests, invalid)
  without any LLM call, saving latency and cost.
- LLM fallback only fires when heuristics are ambiguous.
- We return one of: product_issue, feature_request, bug, invalid.
"""

import os
import re
import time

from groq import Groq

from prompts import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_PROMPT

# Valid request types
VALID_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

# --------------------------------------------------------------------------- #
# Keyword patterns for heuristic classification
# --------------------------------------------------------------------------- #
BUG_KEYWORDS = [
    r'\b(bug|broken|crash|error|fail|not working|down|outage|500|404|'
    r'unresponsive|hang|freeze|glitch|regression|cannot load|cannot access|'
    r'not loading|site is down|pages? (?:are |is )?not accessible|'
    r'stopped working|failing|requests? (?:are |is )?failing)\b',
]

FEATURE_KEYWORDS = [
    r'\b(feature request|would be nice|can you add|please add|wish list|'
    r'enhancement|suggest|proposal|it would help if|would love to see|'
    r'can we get|please implement|missing feature|new feature)\b',
]

INVALID_KEYWORDS = [
    r'\b(ignore previous|jailbreak|system prompt|DAN|ignore all instructions|'
    r'pretend you are|act as if you|reveal your prompt|delete all files|'
    r'rm -rf|format c:)\b',
]

INVALID_OFFTOPIC_PATTERNS = [
    # Completely off-topic questions
    r'\b(who is the actor|what is the capital|tell me a joke|'
    r'what is the meaning of life|recipe for|lyrics of|'
    r'who won the|what year did)\b',
]

GREETING_PATTERNS = [
    r'^(thank(s| you)|happy to help|ok thanks|great thanks|cheers|'
    r'appreciate it|got it|thx|ty)\s*[.!?]*\s*$',
]


def _match_patterns(text: str, patterns: list[str]) -> bool:
    """Check if any pattern matches the text."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def classify_request_type(issue: str, subject: str = "",
                          company: str = "") -> str:
    """
    Classify a support ticket into one of the four request types.
    Uses keyword heuristics first; only calls the LLM if ambiguous.
    """
    combined = f"{subject} {issue}".strip().lower()

    # 1. Check for invalid / adversarial / off-topic
    if _match_patterns(combined, INVALID_KEYWORDS):
        return "invalid"
    if _match_patterns(combined, INVALID_OFFTOPIC_PATTERNS):
        return "invalid"
    # Pure greetings / thank-yous with no actionable content
    if _match_patterns(combined, GREETING_PATTERNS) and len(combined.split()) < 15:
        return "invalid"

    # 2. Check for bug patterns
    bug_match = _match_patterns(combined, BUG_KEYWORDS)

    # 3. Check for feature request patterns
    feature_match = _match_patterns(combined, FEATURE_KEYWORDS)

    # If exactly one matches, return it
    if bug_match and not feature_match:
        return "bug"
    if feature_match and not bug_match:
        return "feature_request"

    # 4. If both or neither match, or ambiguous — use LLM
    if bug_match and feature_match:
        # Ambiguous — let LLM decide
        return _llm_classify(issue, subject, company)

    # Neither matched — most tickets are product_issue, but let LLM decide
    # for better accuracy. If LLM fails, default to product_issue.
    return _llm_classify(issue, subject, company)


def _llm_classify(issue: str, subject: str, company: str,
                  max_retries: int = 3) -> str:
    """Fallback: use LLM for classification when heuristics are ambiguous.
    Retries with exponential backoff on connection errors."""
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        return "product_issue"  # safe default

    client = Groq(api_key=api_key)
    user_msg = CLASSIFICATION_USER_PROMPT.format(
        issue=issue, subject=subject, company=company
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=20,
                temperature=0,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )

            result = response.choices[0].message.content.strip().lower()

            # Validate
            if result in VALID_TYPES:
                return result

            # Try to extract a valid type from the response
            for vt in VALID_TYPES:
                if vt in result:
                    return vt

            return "product_issue"  # safe default
        except Exception as e:
            wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
            print(f"[classifier] LLM attempt {attempt+1}/{max_retries} failed: {e}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    print(f"[classifier] All {max_retries} retries exhausted, defaulting to product_issue")
    return "product_issue"  # safe default after all retries
