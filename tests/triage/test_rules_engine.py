"""
Unit tests for the deterministic triage rules engine.

No DB or network required — all functions are pure keyword-matching logic.
"""

import types

import pytest

from src.db.models import Gender, LifeStage, Species, TriageLevel
from src.triage.rules_engine import (
    TriageRuleResult,
    classify_by_rules,
    compare_and_resolve,
    detect_triage_from_response,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pet(
    species: Species = Species.CAT,
    gender: Gender = Gender.FEMALE,
    stage: LifeStage = LifeStage.ADULT,
) -> types.SimpleNamespace:
    """Build a minimal Pet stub (no DB or SQLAlchemy session required)."""
    return types.SimpleNamespace(species=species, gender=gender, stage=stage)


# ── RED keyword triggers ──────────────────────────────────────────────────────


def test_cant_breathe_is_red() -> None:
    result = classify_by_rules(None, "my cat can't breathe")
    assert result.classification == TriageLevel.RED


def test_seizure_is_red() -> None:
    result = classify_by_rules(None, "she's having a seizure right now")
    assert result.classification == TriageLevel.RED


def test_collapsed_is_red() -> None:
    result = classify_by_rules(None, "he just collapsed and won't wake up")
    assert result.classification == TriageLevel.RED


def test_toxin_chocolate_is_red() -> None:
    result = classify_by_rules(None, "my dog ate chocolate")
    assert result.classification == TriageLevel.RED


def test_heavy_bleeding_is_red() -> None:
    result = classify_by_rules(None, "heavy bleeding from the leg")
    assert result.classification == TriageLevel.RED


# ── ORANGE keyword triggers ───────────────────────────────────────────────────


def test_vomiting_is_orange() -> None:
    result = classify_by_rules(None, "dog has been vomiting since morning")
    assert result.classification == TriageLevel.ORANGE


def test_not_eating_is_orange() -> None:
    result = classify_by_rules(None, "cat is not eating today")
    assert result.classification == TriageLevel.ORANGE


def test_limping_is_orange() -> None:
    result = classify_by_rules(None, "she's limping on her back leg")
    assert result.classification == TriageLevel.ORANGE


def test_diarrhea_is_orange() -> None:
    result = classify_by_rules(None, "my dog has diarrhea")
    assert result.classification == TriageLevel.ORANGE


# ── GREEN (no trigger) ────────────────────────────────────────────────────────


def test_normal_question_is_green() -> None:
    result = classify_by_rules(None, "how often should I feed my cat?")
    assert result.classification == TriageLevel.GREEN


def test_grooming_question_is_green() -> None:
    result = classify_by_rules(None, "when should I brush my dog?")
    assert result.classification == TriageLevel.GREEN


# ── RED combination triggers ──────────────────────────────────────────────────


def test_vomit_lethargy_anorexia_combo_is_orange() -> None:
    """Vomit + lethargy + not eating stays ORANGE without a life-threatening
    signal (breathing, blood, collapse). Each is an individual ORANGE trigger."""
    result = classify_by_rules(
        None, "vomiting, very lethargic, and not eating at all"
    )
    assert result.classification == TriageLevel.ORANGE


def test_bloody_diarrhea_lethargy_is_red() -> None:
    result = classify_by_rules(
        None, "bloody diarrhea and very lethargic since yesterday"
    )
    assert result.classification == TriageLevel.RED
    assert "combo:bloody_diarrhea_lethargy" in result.matched_rules


# ── Pet-specific triggers ─────────────────────────────────────────────────────


def test_male_cat_urinary_is_red() -> None:
    pet = _pet(species=Species.CAT, gender=Gender.MALE)
    result = classify_by_rules(pet, "he's straining in the litter box")
    assert result.classification == TriageLevel.RED
    assert "pet:male_cat_urinary_blockage" in result.matched_rules


def test_female_cat_urinary_is_not_red() -> None:
    """Female cats don't get the blockage escalation."""
    pet = _pet(species=Species.CAT, gender=Gender.FEMALE)
    result = classify_by_rules(pet, "she's straining in the litter box")
    # No RED-level keyword, so should be ORANGE or GREEN
    assert result.classification != TriageLevel.RED


def test_kitten_not_eating_is_red() -> None:
    pet = _pet(stage=LifeStage.KITTEN)
    result = classify_by_rules(pet, "kitten not eating anything")
    assert result.classification == TriageLevel.RED
    assert "pet:young_animal_anorexia" in result.matched_rules


def test_senior_pet_orange_escalates_confidence() -> None:
    pet = _pet(stage=LifeStage.SENIOR)
    result = classify_by_rules(pet, "dog has been vomiting")
    assert result.classification == TriageLevel.ORANGE
    assert result.confidence == 0.8  # escalated for vulnerable stage


# ── detect_triage_from_response ───────────────────────────────────────────────


def test_response_with_emergency_signal_is_red() -> None:
    text = "This is an emergency — please go to the vet immediately."
    level = detect_triage_from_response(text)
    assert level == TriageLevel.RED


def test_response_with_monitor_signal_is_orange() -> None:
    text = "Watch closely for any changes and see your vet if it gets worse."
    level = detect_triage_from_response(text)
    assert level == TriageLevel.ORANGE


def test_response_with_no_signal_is_green() -> None:
    text = "Feeding twice a day is usually fine for adult cats."
    level = detect_triage_from_response(text)
    assert level == TriageLevel.GREEN


# ── compare_and_resolve ───────────────────────────────────────────────────────


def test_rule_stricter_than_llm_overrides() -> None:
    resolved = compare_and_resolve(
        llm_triage=TriageLevel.GREEN,
        rule_classification=TriageLevel.RED,
    )
    assert resolved.final_classification == TriageLevel.RED
    assert resolved.overridden is True
    assert resolved.override_direction == "rules_stricter"


def test_llm_stricter_than_rule_overrides() -> None:
    resolved = compare_and_resolve(
        llm_triage=TriageLevel.RED,
        rule_classification=TriageLevel.ORANGE,
    )
    assert resolved.final_classification == TriageLevel.RED
    assert resolved.overridden is True
    assert resolved.override_direction == "llm_stricter"


def test_agreement_no_override() -> None:
    resolved = compare_and_resolve(
        llm_triage=TriageLevel.ORANGE,
        rule_classification=TriageLevel.ORANGE,
    )
    assert resolved.final_classification == TriageLevel.ORANGE
    assert resolved.overridden is False


def test_none_llm_uses_rule() -> None:
    resolved = compare_and_resolve(
        llm_triage=None,
        rule_classification=TriageLevel.ORANGE,
    )
    assert resolved.final_classification == TriageLevel.ORANGE
    assert resolved.overridden is False
