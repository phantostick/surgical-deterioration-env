"""
Core environment logic.
Implements reset(), step(), and state() following OpenEnv spec.
"""

import random
from typing import Dict, List

from models import (
    AgentAction, ActionType, DeteriorationStage,
    EpisodeState, PatientInfo, ResetRequest,
    StepResult, WardObservation, GraderResult
)
from simulation import generate_patient, advance_patient


TASK_CONFIGS = {
    "task1_single_patient_escalation": {
        "n_patients": 1,
        "n_deteriorating": 1,
        "max_steps": 8,
        "deterioration_speeds": [1.0],
        "rapid_response_cooldown": 0,
        "description": "Single patient, obvious deterioration, 4 hours"
    },
    "task2_subtle_deterioration": {
        "n_patients": 5,
        "n_deteriorating": 1,
        "max_steps": 24,
        "deterioration_speeds": [0.5],
        "rapid_response_cooldown": 2,
        "description": "5 patients, one deteriorating subtly, 12 hours"
    },
    "task3_ward_triage": {
        "n_patients": 20,
        "n_deteriorating": 3,
        "max_steps": 24,
        "deterioration_speeds": [1.5, 0.8, 0.5],
        "rapid_response_cooldown": 2,
        "description": "20 patients, 3 deteriorating at different rates, 12 hours"
    }
}


class SurgicalDeteriorationEnv:

    def __init__(self):
        self._reset_internal()

    def _reset_internal(self):
        self.task_id = "task1_single_patient_escalation"
        self.seed = 42
        self.step_count = 0
        self.max_steps = 8
        self.done = False
        self.total_reward = 0.0
        self.patients: List[PatientInfo] = []
        self.hidden_states: List[dict] = []
        self.rng = random.Random(42)
        self.rapid_response_cooldown = 0
        self.rapid_response_cooldown_steps = 0
        self.false_alarm_count = 0
        self.credibility_score = 1.0
        self.lives_saved = 0
        self.deteriorations_caught_early = 0
        self.patients_coded = 0
        self.trajectory: List[dict] = []

    def reset(self, request: ResetRequest) -> WardObservation:
        self._reset_internal()
        self.task_id = request.task_id
        self.seed = request.seed
        self.rng = random.Random(request.seed)

        config = TASK_CONFIGS.get(request.task_id)
        if config is None:
            raise ValueError(f"Unknown task_id: {request.task_id}")

        self.max_steps = config["max_steps"]
        self.rapid_response_cooldown = config["rapid_response_cooldown"]

        n_patients = config["n_patients"]
        n_deteriorating = config["n_deteriorating"]
        det_speeds = config["deterioration_speeds"]

        deteriorating_indices = self.rng.sample(range(n_patients), n_deteriorating)
        det_speed_map = {
            idx: det_speeds[i] if i < len(det_speeds) else 1.0
            for i, idx in enumerate(deteriorating_indices)
        }

        for pid in range(n_patients):
            patient, hidden = generate_patient(
                patient_id=pid,
                rng=self.rng,
                deteriorating=pid in deteriorating_indices,
                deterioration_speed=det_speed_map.get(pid, 1.0)
            )
            self.patients.append(patient)
            self.hidden_states.append(hidden)

        return self._build_observation()

    def step(self, action: AgentAction) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        patient_id = action.patient_id
        if patient_id < 0 or patient_id >= len(self.patients):
            raise ValueError(f"Invalid patient_id: {patient_id}")

        action_type = action.action
        if action_type == ActionType.RAPID_RESPONSE and self.rapid_response_cooldown_steps > 0:
            action_type = ActionType.CALL_DOCTOR

        # Snapshot hidden states BEFORE advancing (fix false alarm race condition)
        pre_action_hidden = {i: dict(h) for i, h in enumerate(self.hidden_states)}

        step_total_reward = 0.0
        any_coded = False

        for i, patient in enumerate(self.patients):
            act = action_type if i == patient_id else ActionType.MONITOR
            was_escalated_before = patient.escalated

            updated, new_hidden, reward, coded = advance_patient(
                patient=patient,
                hidden_state=self.hidden_states[i],
                action_taken=act,
                step=self.step_count,
                rng=self.rng
            )
            self.patients[i] = updated
            self.hidden_states[i] = new_hidden
            step_total_reward += reward

            if coded and not was_escalated_before:
                self.patients_coded += 1
                any_coded = True

            # Fix: use was_escalated_before for correct lives_saved tracking
            if new_hidden.get("already_escalated") and not was_escalated_before:
                if new_hidden.get("deteriorating") is False:
                    self.deteriorations_caught_early += 1
                    self.lives_saved += 1

        # Fix: use pre-action snapshot for false alarm check
        if action_type == ActionType.RAPID_RESPONSE:
            snapshot = pre_action_hidden[patient_id]
            if not snapshot["deteriorating"] and not snapshot.get("already_escalated"):
                self.false_alarm_count += 1
                self.credibility_score = max(0.1, self.credibility_score - 0.15)
                step_total_reward -= 0.2
            self.rapid_response_cooldown_steps = self.rapid_response_cooldown

        if self.rapid_response_cooldown_steps > 0:
            self.rapid_response_cooldown_steps -= 1

        target = self.patients[patient_id]
        if action_type == ActionType.MONITOR and target.deterioration_stage == DeteriorationStage.STABLE:
            step_total_reward += 0.02

        self.total_reward += step_total_reward
        self.step_count += 1

        self.trajectory.append({
            "step": self.step_count,
            "action": action.dict(),
            "news2_scores": [p.vitals.news2_score for p in self.patients],
            "stages": [p.deterioration_stage for p in self.patients],
            "reward": step_total_reward,
            "false_alarms": self.false_alarm_count
        })

        self.done = self.step_count >= self.max_steps

        return StepResult(
            observation=self._build_observation(),
            reward=step_total_reward,
            done=self.done,
            truncated=self.done and not any_coded,
            info={
                "step": self.step_count,
                "total_reward": self.total_reward,
                "patients_coded": self.patients_coded,
                "false_alarms": self.false_alarm_count
            }
        )

    def state(self) -> EpisodeState:
        return EpisodeState(
            task_id=self.task_id,
            seed=self.seed,
            current_step=self.step_count,
            max_steps=self.max_steps,
            total_reward=self.total_reward,
            done=self.done,
            lives_saved=self.lives_saved,
            deteriorations_caught_early=self.deteriorations_caught_early,
            false_alarms=self.false_alarm_count,
            patients_coded=self.patients_coded
        )

    def grade(self) -> GraderResult:
        if self.task_id == "task1_single_patient_escalation":
            return self._grade_task1()
        elif self.task_id == "task2_subtle_deterioration":
            return self._grade_task2()
        elif self.task_id == "task3_ward_triage":
            return self._grade_task3()
        else:
            raise ValueError(f"No grader for task: {self.task_id}")

    def _grade_task1(self) -> GraderResult:
        hidden = self.hidden_states[0]
        escalated = hidden.get("already_escalated", False)
        coded = hidden.get("coded", False)

        escalation_step = None
        for t in self.trajectory:
            if t["action"]["action"] in ["call_doctor", "rapid_response"]:
                escalation_step = t["step"]
                break

        if coded:
            base_score = 0.0
        elif escalated and escalation_step is not None:
            early_bonus = max(0.0, 1.0 - (escalation_step / self.max_steps))
            base_score = 0.5 + 0.5 * early_bonus
        else:
            base_score = 0.1

        score = max(0.0, min(1.0, base_score - self.false_alarm_count * 0.1))

        return GraderResult(
            task_id=self.task_id,
            score=round(score, 4),
            breakdown={
                "base_score": base_score,
                "false_alarm_penalty": self.false_alarm_count * 0.1,
                "escalated": float(escalated),
                "patient_coded": float(coded),
                "escalation_step": float(escalation_step or -1)
            },
            explanation=(
                f"Patient {'coded' if coded else 'escalated' if escalated else 'not escalated'}. "
                f"Escalation at step {escalation_step}. False alarms: {self.false_alarm_count}."
            )
        )

    def _grade_task2(self) -> GraderResult:
        # Fix: use actual deteriorating flag, not fragile severity_score threshold
        det_idx = next(
            (i for i, h in enumerate(self.hidden_states)
             if h.get("deteriorating") or h.get("already_escalated") or h.get("coded")),
            None
        )

        correct_escalation = False
        escalation_step = None
        wrong_escalations = 0

        for t in self.trajectory:
            act = t["action"]["action"]
            pid = t["action"]["patient_id"]
            if act in ["call_doctor", "rapid_response"]:
                if pid == det_idx:
                    if not correct_escalation:
                        correct_escalation = True
                        escalation_step = t["step"]
                else:
                    wrong_escalations += 1

        coded = self.hidden_states[det_idx]["coded"] if det_idx is not None else False

        if coded:
            detection_score = 0.0
        elif correct_escalation and escalation_step is not None:
            detection_score = max(0.3, 1.0 - (escalation_step / self.max_steps) * 0.7)
        else:
            detection_score = 0.0

        false_alarm_penalty = min(0.4, wrong_escalations * 0.1 + self.false_alarm_count * 0.15)
        score = max(0.0, min(1.0, detection_score - false_alarm_penalty))

        return GraderResult(
            task_id=self.task_id,
            score=round(score, 4),
            breakdown={
                "detection_score": detection_score,
                "false_alarm_penalty": false_alarm_penalty,
                "correct_escalation": float(correct_escalation),
                "patient_coded": float(coded),
                "wrong_escalations": float(wrong_escalations)
            },
            explanation=(
                f"Deteriorating patient was {det_idx}. "
                f"Correct escalation: {correct_escalation} at step {escalation_step}. "
                f"Wrong escalations: {wrong_escalations}."
            )
        )

    def _grade_task3(self) -> GraderResult:
        n_det = sum(1 for h in self.hidden_states if h.get("severity_score", 0) > 0.5)
        lives_ratio = self.lives_saved / max(1, n_det)
        false_alarm_rate = self.false_alarm_count / max(1, self.step_count)
        early_bonus = self.deteriorations_caught_early / max(1, n_det)

        raw_score = (
            0.4 * lives_ratio +
            0.3 * (1.0 - min(1.0, false_alarm_rate * 5)) +
            0.3 * early_bonus
        )
        score = max(0.0, min(1.0, raw_score - self.patients_coded * 0.2))

        return GraderResult(
            task_id=self.task_id,
            score=round(score, 4),
            breakdown={
                "lives_ratio": lives_ratio,
                "false_alarm_rate": false_alarm_rate,
                "early_bonus": early_bonus,
                "coded_penalty": self.patients_coded * 0.2,
                "raw_score": raw_score
            },
            explanation=(
                f"Deteriorating: {n_det}. Saved: {self.lives_saved}. "
                f"Coded: {self.patients_coded}. False alarms: {self.false_alarm_count}."
            )
        )

    def _build_observation(self) -> WardObservation:
        return WardObservation(
            patients=self.patients,
            step=self.step_count,
            hours_remaining=(self.max_steps - self.step_count) * 0.5,
            rapid_response_available=self.rapid_response_cooldown_steps == 0,
            rapid_response_cooldown_steps=self.rapid_response_cooldown_steps,
            false_alarm_count=self.false_alarm_count,
            credibility_score=self.credibility_score,
            task_id=self.task_id,
            seed=self.seed
        )