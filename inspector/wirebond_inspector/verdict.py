"""One word per unit: OK, DEFECT or CHECK.

What: DEFECT if the AI found any defect; CHECK if it found none but the unusual-spot checker
raised its alarm on any camera ("something looks odd, a person should look"); OK otherwise.
Why: the operator, the stack light and the Review filters all need the same simple answer,
worked out in one place.
"""
from __future__ import annotations


def unit_verdict(findings: list[dict], shots: list[dict]) -> str:
    if findings:
        return "DEFECT"
    if any(s.get("anomaly_flag") for s in shots):
        return "CHECK"
    return "OK"
