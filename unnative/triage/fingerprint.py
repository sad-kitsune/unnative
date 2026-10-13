"""Shared fingerprint vocabulary.

A fingerprint is a list of *signals* (each an observation with a weight and the
evidence behind it) plus a derived score and a confidence band. The design goal is
honesty: the report shows every signal and its evidence, so a human can see *why*
a protector was named, and confidence is deliberately capped until the Phase 0
corpus lets us wire in the confirmed, distinctive constants (loader class names,
resource paths, symbol conventions) that the plan's §1 marks as "unknown".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Confidence bands. Kept coarse on purpose — a false precision like "0.83" would
# overstate what structural signals alone can tell us before the corpus exists.
NONE = "none"
LOW = "low"
MEDIUM = "medium"
HIGH = "high"

_BAND_ORDER = {NONE: 0, LOW: 1, MEDIUM: 2, HIGH: 3}


def band_max(a: str, b: str) -> str:
    return a if _BAND_ORDER[a] >= _BAND_ORDER[b] else b


@dataclass
class Signal:
    name: str
    present: bool
    weight: float               # contribution to the score when present
    detail: str                 # one line a human can read
    evidence: list = field(default_factory=list)   # concrete items (names, counts)
    # The confidence this single signal would justify *on its own* if present.
    # The fingerprint takes the max over present signals, so one distinctive
    # structural signal (e.g. JVMTP's per-triple libs) can reach HIGH while weak
    # corroborating ones stay LOW.
    confidence_if_present: str = LOW


@dataclass
class Fingerprint:
    protector: str              # "jnic" | "jvmtp"
    score: float
    confidence: str
    signals: list[Signal]
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "protector": self.protector,
            "score": round(self.score, 3),
            "confidence": self.confidence,
            "signals": [asdict(s) for s in self.signals if s.present],
            "absent": [s.name for s in self.signals if not s.present],
            "notes": self.notes,
        }

    @classmethod
    def from_signals(cls, protector: str, signals: list[Signal],
                     notes: list[str] | None = None) -> "Fingerprint":
        score = sum(s.weight for s in signals if s.present)
        conf = NONE
        for s in signals:
            if s.present:
                conf = band_max(conf, s.confidence_if_present)
        return cls(protector=protector, score=score, confidence=conf,
                   signals=signals, notes=notes or [])
