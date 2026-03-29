"""
Pydantic typed models for the Surgical Deterioration Environment.
All observation, action, and result types are defined here.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    MONITOR = "monitor"                  # Do nothing, keep watching
    CALL_DOCTOR = "call_doctor"          # Call the ward doctor
    RAPID_RESPONSE = "rapid_response"   # Activate rapid response team (scarce)
    ORDER_LABS = "order_labs"            # Order urgent blood/lab work


class DeteriorationStage(str, Enum):
    STABLE = "stable"
    WATCH = "watch"           # Mild concern, vitals drifting
    CONCERN = "concern"       # Multiple vitals abnormal
    CRITICAL = "critical"     # Imminent deterioration
    CODED = "coded"           # Full arrest — worst outcome


class SurgeryType(str, Enum):
    CARDIAC = "cardiac"
    ABDOMINAL = "abdominal"
    ORTHOPEDIC = "orthopedic"
    NEUROLOGICAL = "neurological"
    VASCULAR = "vascular"


# ---------------------------------------------------------------------------
# Per-patient vitals
# ---------------------------------------------------------------------------

class PatientVitals(BaseModel):
    heart_rate: float = Field(..., description="Beats per minute. Normal: 60-100")
    systolic_bp: float = Field(..., description="mmHg. Normal: 90-140")
    diastolic_bp: float = Field(..., description="mmHg. Normal: 60-90")
    respiratory_rate: float = Field(..., description="Breaths/min. Normal: 12-20")
    spo2: float = Field(..., description="Oxygen saturation %. Normal: >95")
    temperature: float = Field(..., description="Celsius. Normal: 36.5-37.5")
    urine_output_ml_hr: float = Field(..., description="mL/hr. Normal: >0.5 mL/kg/hr (~35+)")
    gcs: int = Field(..., description="Glasgow Coma Scale 3-15. Normal: 15")
    pain_score: int = Field(..., description="0-10 pain scale")
    news2_score: float = Field(..., description="NEWS2 early warning score 0-20. >7 = high risk")


class PatientInfo(BaseModel):
    patient_id: int
    age: int
    surgery_type: SurgeryType
    hours_post_surgery: float
    vitals: PatientVitals
    deterioration_stage: DeteriorationStage
    nursing_flags: List[str] = Field(
        default_factory=list,
        description="Free-text nursing observations e.g. 'patient confused', 'wound oozing'"
    )
    last_action_taken: Optional[ActionType] = None
    escalated: bool = False


# ---------------------------------------------------------------------------
# Ward-level observation (what the agent sees)
# ---------------------------------------------------------------------------

class WardObservation(BaseModel):
    patients: List[PatientInfo]
    step: int = Field(..., description="Current step in the episode (0-indexed)")
    hours_remaining: float = Field(..., description="Hours left in the shift")
    rapid_response_available: bool = Field(
        ..., description="Whether rapid response team can be activated this step"
    )
    rapid_response_cooldown_steps: int = Field(
        default=0,
        description="Steps until rapid response is available again"
    )
    false_alarm_count: int = Field(
        default=0,
        description="Number of false rapid response activations this episode"
    )
    credibility_score: float = Field(
        default=1.0,
        description="0.0-1.0. Decreases with false alarms. Affects doctor response time."
    )
    task_id: str
    seed: int


# ---------------------------------------------------------------------------
# Agent action
# ---------------------------------------------------------------------------

class AgentAction(BaseModel):
    patient_id: int = Field(..., description="Which patient to act on (0-indexed)")
    action: ActionType = Field(..., description="What action to take")
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional reasoning string (used for LLM agents, ignored by grader)"
    )


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: WardObservation
    reward: float = Field(..., description="Reward for this step. Range: -1.0 to +1.0")
    done: bool = Field(..., description="Whether the episode has ended")
    truncated: bool = Field(default=False, description="Episode ended due to max steps")
    info: Dict = Field(default_factory=dict, description="Extra diagnostic info")


# ---------------------------------------------------------------------------
# Episode state (returned by /state endpoint)
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    task_id: str
    seed: int
    current_step: int
    max_steps: int
    total_reward: float
    done: bool
    lives_saved: int
    deteriorations_caught_early: int
    false_alarms: int
    patients_coded: int


# ---------------------------------------------------------------------------
# Reset request
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(
        default="task1_single_patient_escalation",
        description="Which task to run"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


# ---------------------------------------------------------------------------
# Grader result
# ---------------------------------------------------------------------------

class GraderResult(BaseModel):
    task_id: str
    score: float = Field(..., description="Final score 0.0-1.0")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Score components"
    )
    explanation: str = ""
