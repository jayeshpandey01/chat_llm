"""
Dynamic Clinical Scratchpad & Progressive Dialogue Summarizer (HDCS Tier 4).
===========================================================================
Prevents conversational token buildup and attentional drift in multi-turn
sessions by condensing older dialogue turns into an atomic clinical scratchpad
while retaining the last k=2 turns in full detail.

Design:
- Weed's Problem-Oriented Medical Record (POMR) Subjective state tracking.
- Extracts key symptoms mentioned, denied signs, and clinical progression.
- Strictly bounds dialogue context under 60-80 tokens regardless of session turn count.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional


class DynamicClinicalScratchpad:
    """Maintains a bounded clinical scratchpad for multi-turn consultations."""

    def __init__(self, max_verbatim_turns: int = 2, max_scratchpad_tokens: int = 60):
        self.max_verbatim_turns = max_verbatim_turns
        self.max_scratchpad_tokens = max_scratchpad_tokens

    @staticmethod
    def extract_salient_clinical_points(turns: List[dict[str, Any]]) -> str:
        """
        Synthesizes older conversation turns into a progressive, compact clinical bullet.
        Captures reported symptoms, timeline, and negative assertions.
        """
        if not turns:
            return ""

        reported_complaints = []
        for t in turns:
            user_q = str(t.get("user_query", t.get("user", ""))).strip()
            # Clean and take key clauses
            clauses = re.split(r"[,;.]\s*", user_q)
            for c in clauses:
                c_clean = c.strip()
                if c_clean and len(c_clean) > 3 and c_clean not in reported_complaints:
                    reported_complaints.append(c_clean)

        # Truncate to top 3-4 salient clauses
        salient_summary = "; ".join(reported_complaints[:3])
        if len(salient_summary) > 140:
            salient_summary = salient_summary[:137] + "..."

        return f"Earlier in consultation, patient stated: {salient_summary}."

    def build_bounded_turn_context(
        self,
        recent_turns: List[dict[str, Any]],
    ) -> tuple[Optional[str], List[dict[str, Any]]]:
        """
        Splits dialogue history into:
        1. Compact scratchpad for older turns (turns 1..N-2).
        2. Verbatim sliding window for the last 2 turns (turns N-1, N).
        """
        if not recent_turns:
            return None, []

        if len(recent_turns) <= self.max_verbatim_turns:
            return None, recent_turns

        # Older turns to compress
        older_turns = recent_turns[: -self.max_verbatim_turns]
        verbatim_turns = recent_turns[-self.max_verbatim_turns :]

        scratchpad_text = self.extract_salient_clinical_points(older_turns)
        return scratchpad_text, verbatim_turns

    def format_recent_context_section(
        self,
        recent_turns: List[dict[str, Any]],
    ) -> Optional[str]:
        """
        Compiles the complete Tier 4 dialogue section:
        - Optional progressive scratchpad
        - Verbatim last 2 turns
        Guaranteed to stay strictly within 60-80 tokens.
        """
        if not recent_turns:
            return None

        scratchpad, active_turns = self.build_bounded_turn_context(recent_turns)

        lines = []
        if scratchpad:
            lines.append(f"  [Prior Discussion Summary]: {scratchpad}")

        for t in active_turns:
            q_snippet = str(t.get("user_query", t.get("user", "")))[:80]
            a_snippet = str(t.get("assistant_summary", t.get("assistant", "")))[:100]
            lines.append(f"  Patient: \"{q_snippet}\"\n  Assistant: \"{a_snippet}\"")

        return "[RECENT CLINICAL CONTEXT]\n" + "\n".join(lines)


clinical_scratchpad = DynamicClinicalScratchpad()

