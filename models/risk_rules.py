"""
Rule-based legal risk detection engine — 30-rule comprehensive set.

Rules are grouped into 8 legal categories:
  Liability | Termination | Competition & IP | Payment |
  Dispute Resolution | Data & Confidentiality | Enforceability
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

    # (regex_pattern, risk_level, category, explanation)
    _RULE_DEFINITIONS: List[tuple] = [

        # ── LIABILITY ──────────────────────────────────────────────────────────
        (r"unlimited liability",             "High",   "Liability",
         "Imposes unlimited financial liability on one party."),
        (r"waiver of liability",             "High",   "Liability",
         "Waives all liability — may be unenforceable in some jurisdictions."),
        (r"indemnif\w*.*hold harmless",      "High",   "Liability",
         "Broad indemnification + hold-harmless clause creates unlimited financial exposure."),
        (r"indemnif\w+",                     "Medium", "Liability",
         "Indemnification clause — review scope and caps carefully."),
        (r"limitation of liability",         "Medium", "Liability",
         "Liability cap clause — verify the cap amount is sufficient for your exposure."),
        (r"consequential\s+damages",         "Medium", "Liability",
         "Exclusion/inclusion of consequential damages significantly changes risk exposure."),
        (r"liquidated damages",              "Medium", "Liability",
         "Pre-determined penalty could result in significant financial loss."),

        # ── TERMINATION ────────────────────────────────────────────────────────
        (r"termination without (cause|notice)", "High", "Termination",
         "Allows termination without cause or prior notice — sudden financial exposure."),
        (r"immediate(ly)?\s+terminat\w+",    "High",   "Termination",
         "Permits immediate termination with no cure or notice period."),
        (r"termination for convenience",     "Medium", "Termination",
         "Either party can exit at will — no guaranteed contract duration."),
        (r"automatic(ally)?\s+(renew|renewal)", "Medium", "Termination",
         "Auto-renewal may lock parties into unintended obligations."),
        (r"notice period",                   "Low",    "Termination",
         "Review notice period length — short windows may be insufficient for transition."),

        # ── COMPETITION & IP ───────────────────────────────────────────────────
        (r"non.?compet\w+",                  "High",   "Competition",
         "Non-compete clause restricts future employment or business activities."),
        (r"non.?solicit\w*",                 "High",   "Competition",
         "Non-solicitation clause restricts future recruitment or client engagement."),
        (r"exclusive\s+(right|license|agreement)", "High", "IP",
         "Exclusivity grant — prevents you from working with other parties."),
        (r"assigns?\s+all\s+(intellectual property|ip\b|rights)", "High", "IP",
         "Broad IP assignment — you may lose ownership of all work product."),
        (r"work.?for.?hire",                 "High",   "IP",
         "Work-for-hire clause transfers all IP ownership to the commissioning party."),
        (r"perpetual\s+(license|right)",     "Medium", "IP",
         "Perpetual license — ensure scope is limited and revocable conditions exist."),
        (r"royalt(y|ies)",                   "Low",    "IP",
         "Royalty clause — confirm rates, calculation method, and audit rights."),

        # ── PAYMENT & FINANCE ──────────────────────────────────────────────────
        (r"interest\s+on\s+(late|overdue)\s+payment", "Medium", "Payment",
         "Late payment interest may compound rapidly — verify the applicable rate."),
        (r"penalty\s+(clause|fee)",          "Medium", "Payment",
         "Penalty clause imposes financial sanctions beyond actual loss suffered."),
        (r"set.?off",                        "Medium", "Payment",
         "Right of set-off allows one party to deduct amounts from payments owed."),
        (r"payment\s+in\s+advance",          "Low",    "Payment",
         "Upfront payment required — review refund terms if project is cancelled."),

        # ── DISPUTE RESOLUTION ─────────────────────────────────────────────────
        (r"binding\s+arbitration",           "High",   "Dispute",
         "Binding arbitration removes the right to court trial or jury."),
        (r"class\s+action\s+waiver",         "High",   "Dispute",
         "Waives right to participate in class-action suits."),
        (r"governing law.*arbitration|arbitration.*governing law", "Medium", "Dispute",
         "Foreign governing law + arbitration may significantly increase dispute costs."),
        (r"sole\s+discretion",               "Medium", "Dispute",
         "Grants unilateral and unchecked decision-making power to one party."),

        # ── DATA & CONFIDENTIALITY ─────────────────────────────────────────────
        (r"data\s+breach",                   "High",   "Data Privacy",
         "Data breach provision — verify notification timelines and liability allocation."),
        (r"personal\s+data|personally\s+identifiable", "Medium", "Data Privacy",
         "Personal data handling — confirm GDPR/DPDP Act compliance obligations."),
        (r"confidential(ity)?\s+(agreement|clause|obligation)", "Medium", "Confidentiality",
         "Confidentiality obligation — review scope, duration, and exceptions."),

        # ── ENFORCEABILITY ─────────────────────────────────────────────────────
        (r"no\s+obligation",                 "Low",    "Enforceability",
         "Removes obligation — may affect enforceability of the entire agreement."),
        (r"\bas\s+is\b",                     "Low",    "Enforceability",
         "'As is' clause disclaims all warranties on the subject matter."),
        (r"best\s+efforts",                  "Low",    "Enforceability",
         "'Best efforts' is vague and may be difficult to enforce in court."),
        (r"reasonable\s+efforts",            "Low",    "Enforceability",
         "'Reasonable efforts' is subjective — define measurable obligations instead."),
        (r"force\s+majeure",                 "Low",    "Enforceability",
         "Force majeure clause — review what events are covered and required notice."),
        (r"material\s+(breach|default)",     "Medium", "Enforceability",
         "'Material breach' standard is subjective — disputes often arise over its definition."),
    ]

    def __init__(self) -> None:
        # Pre-compile all patterns with IGNORECASE for efficient reuse
        self.rules: List[Dict[str, Any]] = [
            {
                "pattern":    re.compile(pattern, re.IGNORECASE),
                "risk_level": level,
                "category":   category,
                "reason":     reason,
            }
            for pattern, level, category, reason in self._RULE_DEFINITIONS
        ]
        logger.info("RiskRuleEngine initialised with %d rules.", len(self.rules))

    def analyze_clauses(self, clauses: List[str]) -> List[Dict[str, Any]]:
        """
        Scan each clause against all pre-compiled rules.

        Returns
        -------
        list of dict
            {clause, risk_level, category, explanation, source}
        """
        findings: List[Dict[str, Any]] = []

        for clause in clauses:
            if not clause.strip():
                continue
            for rule in self.rules:
                if rule["pattern"].search(clause):
                    findings.append({
                        "clause":      clause,
                        "risk_level":  rule["risk_level"],
                        "category":    rule["category"],
                        "explanation": rule["reason"],
                        "source":      "rule_based",
                    })

        logger.debug(
            "Rule engine found %d risk(s) across %d clause(s).",
            len(findings), len(clauses),
        )
        return findings
