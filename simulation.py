"""
Seeded patient simulation engine.
Given the same seed, produces identical patient trajectories every time.
This is what makes graders deterministic and reproducible.
"""

import random
import math
from typing import List, Tuple
from models import (
    PatientInfo, PatientVitals, DeteriorationStage,
    SurgeryType, ActionType
)


# ---------------------------------------------------------------------------
# NEWS2 Score Calculator (real clinical formula)
# ---------------------------------------------------------------------------

def calculate_news2(vitals: PatientVitals) -> float:
    """
    Calculates National Early Warning Score 2 (NEWS2).
    This is the real formula used in NHS hospitals.
    Score > 7 = high risk, requires urgent response.
    """
    score = 0

    # Respiratory rate
    rr = vitals.respiratory_rate
    if rr <= 8 or rr >= 25:
        score += 3
    elif rr >= 21:
        score += 2
    elif rr <= 11:
        score += 1

    # SpO2
    spo2 = vitals.spo2
    if spo2 <= 91:
        score += 3
    elif spo2 <= 93:
        score += 2
    elif spo2 <= 95:
        score += 1

    # Systolic BP
    sbp = vitals.systolic_bp
    if sbp <= 90 or sbp >= 220:
        score += 3
    elif sbp <= 100:
        score += 2
    elif sbp <= 110:
        score += 1

    # Heart rate
    hr = vitals.heart_rate
    if hr <= 40 or hr >= 131:
        score += 3
    elif hr >= 111:
        score += 2
    elif hr <= 50 or hr >= 91:
        score += 1

    # Temperature
    temp = vitals.temperature
    if temp <= 35.0:
        score += 3
    elif temp >= 39.1:
        score += 2
    elif temp <= 36.0 or temp >= 38.1:
        score += 1

    # GCS (consciousness)
    if vitals.gcs <= 14:
        score += 3

    # Urine output (low output = concern)
    if vitals.urine_output_ml_hr < 20:
        score += 2
    elif vitals.urine_output_ml_hr < 35:
        score += 1

    return float(score)


# ---------------------------------------------------------------------------
# Patient Generator
# ---------------------------------------------------------------------------

def generate_patient(
    patient_id: int,
    rng: random.Random,
    deteriorating: bool = False,
    deterioration_speed: float = 1.0,
    surgery_type: SurgeryType = None
) -> Tuple[PatientInfo, dict]:
    """
    Generates a patient with initial vitals.
    Returns (PatientInfo, hidden_state) where hidden_state drives future drift.
    """
    if surgery_type is None:
        surgery_type = rng.choice(list(SurgeryType))

    age = rng.randint(35, 80)
    hours_post_surgery = rng.uniform(2, 24)

    # Base vitals — slightly noisy around normal
    hr = rng.gauss(78, 8)
    sbp = rng.gauss(118, 10)
    dbp = rng.gauss(74, 6)
    rr = rng.gauss(15, 2)
    spo2 = rng.gauss(97.5, 0.8)
    temp = rng.gauss(37.0, 0.3)
    urine = rng.gauss(45, 10)
    gcs = 15 if not deteriorating else rng.choice([15, 15, 15, 14])
    pain = rng.randint(2, 5)

    vitals = PatientVitals(
        heart_rate=round(max(50, min(130, hr)), 1),
        systolic_bp=round(max(85, min(160, sbp)), 1),
        diastolic_bp=round(max(50, min(100, dbp)), 1),
        respiratory_rate=round(max(10, min(22, rr)), 1),
        spo2=round(max(90, min(100, spo2)), 1),
        temperature=round(max(35.5, min(39.5, temp)), 1),
        urine_output_ml_hr=round(max(10, min(80, urine)), 1),
        gcs=gcs,
        pain_score=pain,
        news2_score=0.0  # will be calculated below
    )
    vitals.news2_score = calculate_news2(vitals)

    nursing_flags = []
    if deteriorating and rng.random() < 0.4:
        nursing_flags.append(rng.choice([
            "patient seems less responsive than earlier",
            "wound site looks slightly inflamed",
            "patient complained of increased breathlessness",
            "IV site showing mild redness"
        ]))

    # Hidden state drives deterministic deterioration
    hidden_state = {
        "deteriorating": deteriorating,
        "deterioration_speed": deterioration_speed,
        "severity_score": rng.uniform(0.6, 0.9) if deteriorating else rng.uniform(0.0, 0.2),
        "deterioration_type": rng.choice(["sepsis", "cardiac", "respiratory", "hemorrhage"]),
        "steps_until_critical": int(rng.uniform(6, 16) / deterioration_speed) if deteriorating else 999,
        "already_escalated": False,
        "coded": False,
    }

    patient = PatientInfo(
        patient_id=patient_id,
        age=age,
        surgery_type=surgery_type,
        hours_post_surgery=hours_post_surgery,
        vitals=vitals,
        deterioration_stage=DeteriorationStage.STABLE,
        nursing_flags=nursing_flags
    )

    return patient, hidden_state


# ---------------------------------------------------------------------------
# Vitals Drift Engine
# ---------------------------------------------------------------------------

def advance_patient(
    patient: PatientInfo,
    hidden_state: dict,
    action_taken: ActionType,
    step: int,
    rng: random.Random
) -> Tuple[PatientInfo, dict, float, bool]:
    """
    Advances a patient's vitals by one 30-minute step.
    Returns (updated_patient, updated_hidden_state, step_reward, patient_coded)
    """
    det = hidden_state["deteriorating"]
    speed = hidden_state["deterioration_speed"]
    steps_until_critical = hidden_state["steps_until_critical"]
    det_type = hidden_state["deterioration_type"]
    coded = hidden_state["coded"]

    step_reward = 0.0

    # If already coded, nothing changes
    if coded:
        return patient, hidden_state, -0.5, True

    v = patient.vitals

    # --- Apply action effects ---
    if action_taken == ActionType.RAPID_RESPONSE and det:
        # Rapid response halts deterioration
        hidden_state["deteriorating"] = False
        hidden_state["steps_until_critical"] = 999
        hidden_state["already_escalated"] = True
        step_reward += 0.8
    elif action_taken == ActionType.CALL_DOCTOR and det:
        # Doctor slows deterioration
        hidden_state["steps_until_critical"] = steps_until_critical + 3
        hidden_state["already_escalated"] = True
        step_reward += 0.3
    elif action_taken == ActionType.ORDER_LABS and det:
        # Labs give insight but don't treat
        step_reward += 0.1
    elif action_taken == ActionType.RAPID_RESPONSE and not det:
        # False alarm — penalize
        step_reward -= 0.3

    # --- Drift vitals if deteriorating ---
    if hidden_state["deteriorating"]:
        progress = max(0, 1.0 - (steps_until_critical / 12.0))
        noise = lambda s: rng.gauss(0, s)

        if det_type == "sepsis":
            v = PatientVitals(
                heart_rate=round(v.heart_rate + speed * (1.5 + progress * 3) + noise(1.5), 1),
                systolic_bp=round(v.systolic_bp - speed * (1.0 + progress * 2) + noise(2), 1),
                diastolic_bp=round(v.diastolic_bp - speed * 0.5 + noise(1), 1),
                respiratory_rate=round(v.respiratory_rate + speed * (0.5 + progress * 1.5) + noise(0.5), 1),
                spo2=round(v.spo2 - speed * (0.2 + progress * 0.8) + noise(0.3), 1),
                temperature=round(v.temperature + speed * (0.1 + progress * 0.3) + noise(0.1), 1),
                urine_output_ml_hr=round(max(5, v.urine_output_ml_hr - speed * (2 + progress * 5) + noise(2)), 1),
                gcs=max(3, v.gcs - (1 if progress > 0.7 and rng.random() < 0.3 else 0)),
                pain_score=min(10, v.pain_score + (1 if rng.random() < 0.2 else 0)),
                news2_score=0.0
            )
        elif det_type == "cardiac":
            v = PatientVitals(
                heart_rate=round(v.heart_rate + speed * (2 + progress * 4) + noise(2), 1),
                systolic_bp=round(v.systolic_bp - speed * (1.5 + progress * 3) + noise(2), 1),
                diastolic_bp=round(v.diastolic_bp - speed * 1.0 + noise(1), 1),
                respiratory_rate=round(v.respiratory_rate + speed * (1 + progress * 2) + noise(1), 1),
                spo2=round(v.spo2 - speed * (0.5 + progress * 1.5) + noise(0.4), 1),
                temperature=round(v.temperature + noise(0.1), 1),
                urine_output_ml_hr=round(max(5, v.urine_output_ml_hr - speed * 3 + noise(2)), 1),
                gcs=max(3, v.gcs - (1 if progress > 0.6 and rng.random() < 0.4 else 0)),
                pain_score=min(10, v.pain_score + (2 if rng.random() < 0.3 else 0)),
                news2_score=0.0
            )
        elif det_type == "respiratory":
            v = PatientVitals(
                heart_rate=round(v.heart_rate + speed * (1 + progress * 2) + noise(1), 1),
                systolic_bp=round(v.systolic_bp + noise(2), 1),
                diastolic_bp=round(v.diastolic_bp + noise(1), 1),
                respiratory_rate=round(v.respiratory_rate + speed * (2 + progress * 4) + noise(1), 1),
                spo2=round(v.spo2 - speed * (1 + progress * 3) + noise(0.5), 1),
                temperature=round(v.temperature + speed * 0.2 + noise(0.1), 1),
                urine_output_ml_hr=round(max(5, v.urine_output_ml_hr - speed * 1 + noise(2)), 1),
                gcs=max(3, v.gcs - (1 if progress > 0.8 and rng.random() < 0.3 else 0)),
                pain_score=min(10, v.pain_score + (1 if rng.random() < 0.25 else 0)),
                news2_score=0.0
            )
        else:  # hemorrhage
            v = PatientVitals(
                heart_rate=round(v.heart_rate + speed * (2 + progress * 5) + noise(2), 1),
                systolic_bp=round(v.systolic_bp - speed * (2 + progress * 4) + noise(2), 1),
                diastolic_bp=round(v.diastolic_bp - speed * (1 + progress * 2) + noise(1), 1),
                respiratory_rate=round(v.respiratory_rate + speed * (1 + progress * 2) + noise(1), 1),
                spo2=round(v.spo2 - speed * (0.3 + progress * 1) + noise(0.3), 1),
                temperature=round(v.temperature - speed * 0.2 + noise(0.1), 1),
                urine_output_ml_hr=round(max(5, v.urine_output_ml_hr - speed * (3 + progress * 8) + noise(3)), 1),
                gcs=max(3, v.gcs - (1 if progress > 0.5 and rng.random() < 0.4 else 0)),
                pain_score=min(10, v.pain_score + (1 if rng.random() < 0.3 else 0)),
                news2_score=0.0
            )

        # Clamp vitals to physiological ranges
        v = _clamp_vitals(v)
        v.news2_score = calculate_news2(v)
        hidden_state["steps_until_critical"] = max(0, steps_until_critical - 1)

        # Determine deterioration stage
        if v.news2_score >= 7 or hidden_state["steps_until_critical"] == 0:
            stage = DeteriorationStage.CRITICAL
        elif v.news2_score >= 5:
            stage = DeteriorationStage.CONCERN
        elif v.news2_score >= 3:
            stage = DeteriorationStage.WATCH
        else:
            stage = DeteriorationStage.STABLE

        # Check if patient codes
        patient_coded = False
        if hidden_state["steps_until_critical"] == 0 and not hidden_state["already_escalated"]:
            hidden_state["coded"] = True
            patient_coded = True
            stage = DeteriorationStage.CODED
            step_reward -= 1.0

    else:
        # Stable patient — minor random noise only
        v = PatientVitals(
            heart_rate=round(v.heart_rate + rng.gauss(0, 1.5), 1),
            systolic_bp=round(v.systolic_bp + rng.gauss(0, 2), 1),
            diastolic_bp=round(v.diastolic_bp + rng.gauss(0, 1), 1),
            respiratory_rate=round(v.respiratory_rate + rng.gauss(0, 0.5), 1),
            spo2=round(v.spo2 + rng.gauss(0, 0.3), 1),
            temperature=round(v.temperature + rng.gauss(0, 0.1), 1),
            urine_output_ml_hr=round(max(20, v.urine_output_ml_hr + rng.gauss(0, 3)), 1),
            gcs=v.gcs,
            pain_score=v.pain_score,
            news2_score=0.0
        )
        v = _clamp_vitals(v)
        v.news2_score = calculate_news2(v)
        stage = DeteriorationStage.STABLE
        patient_coded = False

    # Update nursing flags occasionally
    flags = []
    if v.news2_score >= 5:
        flags.append("nurse concerned about patient trajectory")
    if v.urine_output_ml_hr < 25:
        flags.append("low urine output noted")
    if v.gcs < 15:
        flags.append("patient less responsive to verbal stimuli")
    if v.respiratory_rate > 20:
        flags.append("patient breathing faster than normal")

    updated_patient = PatientInfo(
        patient_id=patient.patient_id,
        age=patient.age,
        surgery_type=patient.surgery_type,
        hours_post_surgery=patient.hours_post_surgery + 0.5,
        vitals=v,
        deterioration_stage=stage,
        nursing_flags=flags,
        last_action_taken=action_taken,
        escalated=hidden_state["already_escalated"]
    )

    return updated_patient, hidden_state, step_reward, patient_coded


def _clamp_vitals(v: PatientVitals) -> PatientVitals:
    return PatientVitals(
        heart_rate=round(max(30, min(180, v.heart_rate)), 1),
        systolic_bp=round(max(60, min(200, v.systolic_bp)), 1),
        diastolic_bp=round(max(40, min(120, v.diastolic_bp)), 1),
        respiratory_rate=round(max(6, min(40, v.respiratory_rate)), 1),
        spo2=round(max(70, min(100, v.spo2)), 1),
        temperature=round(max(34.0, min(41.0, v.temperature)), 1),
        urine_output_ml_hr=round(max(0, min(200, v.urine_output_ml_hr)), 1),
        gcs=max(3, min(15, v.gcs)),
        pain_score=max(0, min(10, v.pain_score)),
        news2_score=v.news2_score
    )
