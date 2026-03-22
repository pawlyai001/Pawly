"""
Deterministic keyword + combination-rule triage classifier.

Public API:
    classify_by_rules(pet, description)     -> TriageRuleResult
    detect_triage_from_response(text)       -> TriageLevel | None
    compare_and_resolve(llm, rule)          -> CompareResult

Legacy string helpers (still used by some handlers):
    classify_triage(text)                   -> "RED" | "ORANGE" | "GREEN"
    get_matched_symptoms(text)              -> list[str]

Backward-compat aliases:
    RuleClassificationResult = TriageRuleResult
    ResolveResult            = CompareResult

Keyword lists and combination triggers are module-level constants — edit them
without touching the classifier logic below.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from src.db.models import Gender, LifeStage, Pet, Species, TriageLevel

# ══════════════════════════════════════════════════════════════════════════════
# EDITABLE KEYWORD LISTS
# ══════════════════════════════════════════════════════════════════════════════

RED_KEYWORDS: list[str] = [
    "can't breathe", "cant breathe", "not breathing", "breathing hard",
    "gasping", "labored breathing", "struggling to breathe",
    "seizure", "convulsion", "fitting", "shaking uncontrollably",
    "blood in urine", "can't pee", "cant pee", "straining to urinate",
    "hasn't urinated", "hasnt urinated", "not peeing",
    "ate chocolate", "ate xylitol", "ate lily", "ate antifreeze",
    "ate grape", "ate raisin", "ate onion", "poisoned", "toxic",
    "bloated stomach", "stomach swelling", "distended abdomen",
    "collapsed", "unconscious", "won't wake", "wont wake", "not responding",
    "can't walk", "cant walk", "can't stand", "cant stand",
    "paralyzed", "dragging legs", "hind legs not working",
    "blue gums", "pale gums", "white gums",
    "heavy bleeding", "won't stop bleeding", "wont stop bleeding",
    "eye popping", "eye out", "prolapse",
    "heatstroke", "overheating", "heat exhaustion",
]

ORANGE_KEYWORDS: list[str] = [
    "vomiting", "threw up", "throwing up", "diarrhea", "loose stool",
    "limping", "not eating", "off food", "won't eat", "wont eat",
    "scratching a lot", "excessive scratching", "hair loss", "bald patch",
    "eye discharge", "runny nose", "runny eyes", "coughing",
    "sneezing a lot", "lump", "swelling", "drinking more",
    "peeing more", "weight loss", "weight gain",
]

# Signals that indicate what triage level the LLM response implied
_RED_RESPONSE_SIGNALS: tuple[str, ...] = (
    "urgent", "emergency", "immediately", "emergency vet", "do not wait",
    "life-threatening", "critical", "rush", "go now", "call the vet now",
)
_ORANGE_RESPONSE_SIGNALS: tuple[str, ...] = (
    "watch closely", "monitor", "keep an eye", "concerning", "see your vet",
    "worth checking", "may want to", "could be", "watch for",
)

# Severity ordering for comparison
_SEVERITY: dict[TriageLevel, int] = {
    TriageLevel.GREEN: 0,
    TriageLevel.ORANGE: 1,
    TriageLevel.RED: 2,
}


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class TriageRuleResult:
    """Result of classify_by_rules()."""

    classification: TriageLevel
    matched_rules: list[str] = field(default_factory=list)
    confidence: float = 0.5

    # Backward-compat alias so orchestrator code using .matched_patterns still works
    @property
    def matched_patterns(self) -> list[str]:
        return self.matched_rules


@dataclass
class CompareResult:
    """Result of compare_and_resolve()."""

    final_classification: TriageLevel
    overridden: bool
    override_direction: str = ""   # "" | "rules_stricter" | "llm_stricter"


# Backward-compat aliases
RuleClassificationResult = TriageRuleResult
ResolveResult = CompareResult


# ── Core classifier ───────────────────────────────────────────────────────────


def classify_by_rules(pet: Optional[Pet], description: str) -> TriageRuleResult:
    """
    Classify *description* using keyword matching and combination triggers.

    Pet-aware rules:
    - Male cat + urinary symptoms → RED (potential blockage)
    - Puppy/kitten + not eating   → RED (critical in young animals)
    - Senior/puppy/kitten + ORANGE symptoms → confidence bump

    Returns TriageRuleResult with classification, matched_rules, confidence.
    """
    desc_lower = description.lower()
    matched: list[str] = []

    # ── RED keyword scan ──────────────────────────────────────────────────────
    for kw in RED_KEYWORDS:
        if kw in desc_lower:
            matched.append(f"keyword_red:{kw.replace(' ', '_')}")

    # ── Boolean symptom flags for combination triggers ────────────────────────
    has_vomit       = any(w in desc_lower for w in ["vomit", "threw up", "throwing up"])
    has_lethargy    = any(w in desc_lower for w in ["lethargy", "lethargic", "no energy", "not moving", "very tired"])
    has_not_eating  = any(w in desc_lower for w in ["not eating", "won't eat", "wont eat", "off food", "refuses food"])
    has_diarrhea    = any(w in desc_lower for w in ["diarrhea", "loose stool", "watery stool"])
    has_blood       = any(w in desc_lower for w in ["blood", "bloody", "bleeding"])
    has_breathing   = any(w in desc_lower for w in ["breath", "breathing", "gasping", "panting hard"])
    has_urinary     = any(w in desc_lower for w in ["straining", "urination", "pee", "urine", "litter box", "can't pee", "cant pee"])

    # ── RED combination triggers ──────────────────────────────────────────────
    # Note: vomit + lethargy + not eating alone stays ORANGE — each keyword
    # is already an individual ORANGE trigger. RED requires a life-threatening
    # signal (breathing difficulty, blood, collapse, etc.).

    if has_diarrhea and has_blood and has_lethargy:
        matched.append("combo:bloody_diarrhea_lethargy")

    other_symptoms = [has_vomit, has_lethargy, has_not_eating, has_diarrhea]
    if has_breathing and any(other_symptoms):
        matched.append("combo:breathing_plus_other")

    # ── Pet-specific RED triggers ─────────────────────────────────────────────
    if pet is not None:
        if pet.species == Species.CAT and pet.gender == Gender.MALE and has_urinary:
            matched.append("pet:male_cat_urinary_blockage")

        if pet.stage in (LifeStage.PUPPY, LifeStage.KITTEN) and has_not_eating:
            matched.append("pet:young_animal_anorexia")

    if matched:
        return TriageRuleResult(TriageLevel.RED, matched, confidence=0.95)

    # ── ORANGE keyword scan ───────────────────────────────────────────────────
    orange_matched: list[str] = []
    for kw in ORANGE_KEYWORDS:
        if kw in desc_lower:
            orange_matched.append(f"symptom_orange:{kw.replace(' ', '_')}")

    if orange_matched:
        # Escalate confidence for vulnerable life stages
        is_vulnerable = (
            pet is not None
            and pet.stage in (LifeStage.PUPPY, LifeStage.KITTEN, LifeStage.SENIOR)
        )
        if is_vulnerable:
            orange_matched.append("pet:age_escalation")
        conf = 0.8 if is_vulnerable else 0.7
        return TriageRuleResult(TriageLevel.ORANGE, orange_matched, confidence=conf)

    return TriageRuleResult(TriageLevel.GREEN, [], confidence=0.5)


def detect_triage_from_response(text: str) -> Optional[TriageLevel]:
    """
    Infer the triage level implied by an LLM response by scanning for
    urgency and monitoring signals. Returns None if no clear signal.
    """
    lower = text.lower()
    if any(sig in lower for sig in _RED_RESPONSE_SIGNALS):
        return TriageLevel.RED
    if any(sig in lower for sig in _ORANGE_RESPONSE_SIGNALS):
        return TriageLevel.ORANGE
    return TriageLevel.GREEN


def compare_and_resolve(
    llm_triage: Optional[TriageLevel],
    rule_classification: TriageLevel,
) -> CompareResult:
    """
    Combine LLM-inferred triage with rule-engine result. Always takes the
    stricter of the two. Populates override_direction when they disagree.
    """
    if llm_triage is None:
        return CompareResult(rule_classification, overridden=False, override_direction="")

    llm_sev  = _SEVERITY[llm_triage]
    rule_sev = _SEVERITY[rule_classification]

    if llm_sev == rule_sev:
        return CompareResult(llm_triage, overridden=False, override_direction="")

    if rule_sev > llm_sev:
        return CompareResult(rule_classification, overridden=True, override_direction="rules_stricter")

    return CompareResult(llm_triage, overridden=True, override_direction="llm_stricter")


# ── Legacy helpers ────────────────────────────────────────────────────────────

TriageResult = Literal["RED", "ORANGE", "GREEN"]


def classify_triage(text: str) -> TriageResult:
    """Classify text → 'RED' | 'ORANGE' | 'GREEN' (legacy string form)."""
    result = classify_by_rules(None, text)
    return result.classification.value.upper()  # type: ignore[return-value]


def get_matched_symptoms(text: str) -> list[str]:
    """Return matched rule names for *text* (legacy helper)."""
    return classify_by_rules(None, text).matched_rules
