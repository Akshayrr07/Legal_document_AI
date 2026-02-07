import re
from typing import List, Dict


class RiskRuleEngine:
    """
    Rule-based legal risk detection engine.
    Flags known high-risk legal patterns with explanations.
    """

    def __init__(self):
        self.rules = [
            {
                "pattern": r"termination without notice",
                "risk_level": "High",
                "reason": "Allows contract termination without prior notice."
            },
            {
                "pattern": r"unlimited liability",
                "risk_level": "High",
                "reason": "Imposes unlimited liability on one party."
            },
            {
                "pattern": r"indemnify.*hold harmless",
                "risk_level": "Medium",
                "reason": "Indemnification clause may impose financial risk."
            },
            {
                "pattern": r"sole discretion",
                "risk_level": "Medium",
                "reason": "Grants unilateral decision power to one party."
            },
            {
                "pattern": r"no obligation",
                "risk_level": "Low",
                "reason": "Clause removes obligation, may affect enforceability."
            }
        ]

    def analyze_clauses(self, clauses: List[str]) -> List[Dict]:
        """
        Analyze clauses using predefined legal risk rules.
        """
        findings = []

        for clause in clauses:
            clause_lower = clause.lower()

            for rule in self.rules:
                if re.search(rule["pattern"], clause_lower):
                    findings.append({
                        "clause": clause,
                        "risk_level": rule["risk_level"],
                        "explanation": rule["reason"],
                        "source": "rule_based"
                    })

        return findings
