"""
Rule-based legal risk detection engine.

Improvements over the original:
- Uses re.IGNORECASE flag instead of calling .lower() on every clause (performance)
- Pre-compiles all rule patterns at init time for faster repeated matching
- Expanded rule set with additional common legal risk patterns
- Full type hints
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RiskRuleEngine:
    """
    Rule-based engine that flags known high-risk legal patterns with explanations.
    Patterns are pre-compiled at init time for maximum throughput.
    """

    # Rule definitions: each entry is (regex_pattern, risk_level, explanation)
    _RULE_DEFINITIONS: List[tuple] = [
        (
            r"termination without notice",
            "High",
            "Allows contract termination without prior notice.",
        ),
        (
            r"unlimited liability",
            "High",
            "Imposes unlimited financial liability on one party.",
        ),
        (
            r"indemnif\w*.*hold harmless",
            "High",
            "Broad indemnification clause may impose unlimited financial exposure.",
        ),
        (
            r"waiver of liability",
            "High",
            "Waives all liability — may be unenforceable in some jurisdictions.",
        ),
        (
            r"non.?compet\w+",
            "High",
            "Non-compete clause restricts future employment or business activities.",
        ),
        (
            r"sole discretion",
            "Medium",
            "Grants unilateral decision-making power to one party.",
        ),
        (
            r"automatic renewal",
            "Medium",
            "Contract renews automatically — may create unintended obligations.",
        ),
        (
            r"liquidated damages",
            "Medium",
            "Pre-determined penalty clause could result in significant financial loss.",
        ),
        (
            r"governing law.*arbitration",
            "Medium",
            "Arbitration clause may limit dispute resolution options.",
        ),
        (
            r"no obligation",
            "Low",
            "Clause removes obligation — may affect enforceability of the agreement.",
        ),
        (
            r"as is",
            "Low",
            "'As is' clause disclaims all warranties on the subject matter.",
        ),
        (
            r"best efforts",
            "Low",
            "'Best efforts' standard is vague and may be difficult to enforce.",
        ),
    ]

    def __init__(self) -> None:
        # Pre-compile all patterns with IGNORECASE for efficient reuse
        self.rules: List[Dict[str, Any]] = [
            {
                "pattern": re.compile(pattern, re.IGNORECASE),
                "risk_level": level,
                "reason": reason,
            }
            for pattern, level, reason in self._RULE_DEFINITIONS
        ]
        logger.info("RiskRuleEngine initialised with %d rules.", len(self.rules))

    def analyze_clauses(self, clauses: List[str]) -> List[Dict[str, Any]]:
        """
        Scan each clause against all pre-compiled rules.

        Parameters
        ----------
        clauses : list of str
            Pre-segmented legal clauses from the document.

        Returns
        -------
        list of dict
            Each entry contains: clause, risk_level, explanation, source.
        """
        findings: List[Dict[str, Any]] = []

        for clause in clauses:
            if not clause.strip():
                continue
            for rule in self.rules:
                if rule["pattern"].search(clause):
                    findings.append({
                        "clause": clause,
                        "risk_level": rule["risk_level"],
                        "explanation": rule["reason"],
                        "source": "rule_based",
                    })

        logger.debug("Rule engine found %d risk(s) across %d clause(s).", len(findings), len(clauses))
        return findings
