"""
Baseline inference script for the Surgical Deterioration Environment.
Runs an LLM agent against all 3 tasks and produces reproducible scores.

Usage:
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    export HF_TOKEN="your_token_here"
    python inference.py
"""

import os
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

TASKS = [
    "task1_single_patient_escalation",
    "task2_subtle_deterioration",
    "task3_ward_triage"
]

SEEDS = [42, 123, 777]  # 3 seeds per task for variance check


# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str, seed: int) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed})
    r.raise_for_status()
    return r.json()


def env_step(patient_id: int, action: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={
        "patient_id": patient_id,
        "action": action,
        "reasoning": ""
    })
    r.raise_for_status()
    return r.json()


def env_grade() -> dict:
    r = requests.post(f"{ENV_BASE_URL}/grade")
    r.raise_for_status()
    return r.json()


def env_state() -> dict:
    r = requests.get(f"{ENV_BASE_URL}/state")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# LLM agent prompt builder
# ---------------------------------------------------------------------------

def build_prompt(observation: dict) -> str:
    patients = observation["patients"]
    task_id = observation["task_id"]
    hours_remaining = observation["hours_remaining"]
    rr_available = observation["rapid_response_available"]
    false_alarms = observation["false_alarm_count"]

    patient_summaries = []
    for p in patients:
        v = p["vitals"]
        flags = ", ".join(p["nursing_flags"]) if p["nursing_flags"] else "none"
        summary = (
            f"  Patient {p['patient_id']} | Age {p['age']} | {p['surgery_type']} surgery | "
            f"{p['hours_post_surgery']:.1f}h post-op\n"
            f"    HR={v['heart_rate']} | BP={v['systolic_bp']}/{v['diastolic_bp']} | "
            f"RR={v['respiratory_rate']} | SpO2={v['spo2']}% | Temp={v['temperature']}C | "
            f"Urine={v['urine_output_ml_hr']}mL/hr | GCS={v['gcs']} | NEWS2={v['news2_score']}\n"
            f"    Nursing notes: {flags}"
        )
        patient_summaries.append(summary)

    patients_text = "\n".join(patient_summaries)

    prompt = f"""You are an experienced ICU nurse monitoring a post-surgical ward.

CURRENT WARD STATUS:
Task: {task_id}
Hours remaining in shift: {hours_remaining}
Rapid response team available: {rr_available}
False alarm count this shift: {false_alarms} (too many false alarms reduces credibility)

PATIENTS:
{patients_text}

AVAILABLE ACTIONS:
- monitor: Watch and wait (appropriate for stable patients)
- call_doctor: Call the ward doctor (appropriate for concerning vitals)
- rapid_response: Activate rapid response team (for critical deterioration only - limited use)
- order_labs: Order urgent blood work (appropriate when cause unclear)

NEWS2 SCORE GUIDE:
- 0-2: Low risk, monitor
- 3-4: Low-medium risk, consider calling doctor
- 5-6: Medium risk, call doctor
- 7+: High risk, consider rapid response

TASK: Choose ONE patient and ONE action. Respond in valid JSON only.
Format: {{"patient_id": <int>, "action": "<action>", "reasoning": "<brief reasoning>"}}
"""
    return prompt


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(task_id: str, seed: int) -> dict:
    print(f"\n  Running {task_id} seed={seed}...")

    obs = env_reset(task_id, seed)
    done = False
    steps = 0
    max_steps = {"task1_single_patient_escalation": 8,
                 "task2_subtle_deterioration": 24,
                 "task3_ward_triage": 24}[task_id]

    while not done and steps < max_steps:
        prompt = build_prompt(obs)

        # LLM call
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON action
            # Strip markdown code fences if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action_data = json.loads(raw)
            patient_id = int(action_data["patient_id"])
            action = str(action_data["action"])

            # Validate action
            valid_actions = ["monitor", "call_doctor", "rapid_response", "order_labs"]
            if action not in valid_actions:
                action = "monitor"

            # Clamp patient_id
            n_patients = len(obs["patients"])
            patient_id = max(0, min(n_patients - 1, patient_id))

        except Exception as e:
            print(f"    LLM error at step {steps}: {e}. Defaulting to monitor.")
            patient_id = 0
            action = "monitor"

        # Step environment
        result = env_step(patient_id, action)
        obs = result["observation"]
        done = result["done"]
        steps += 1

        # Small delay to avoid rate limits
        time.sleep(0.3)

    # Grade episode
    grade = env_grade()
    score = grade["score"]
    print(f"    Score: {score:.4f} | {grade['explanation']}")
    return grade


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Surgical Deterioration Environment — Baseline Inference")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")
    print(f"Environment: {ENV_BASE_URL}")

    all_results = {}
    start_time = time.time()

    for task_id in TASKS:
        print(f"\nTask: {task_id}")
        task_scores = []

        for seed in SEEDS:
            grade = run_episode(task_id, seed)
            task_scores.append(grade["score"])

        avg = sum(task_scores) / len(task_scores)
        all_results[task_id] = {
            "scores": task_scores,
            "average": round(avg, 4),
            "seeds": SEEDS
        }
        print(f"  Average score: {avg:.4f}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")

    for task_id, result in all_results.items():
        print(f"{task_id}: {result['average']:.4f} (scores: {result['scores']})")

    overall = sum(r["average"] for r in all_results.values()) / len(all_results)
    print(f"\nOverall average: {overall:.4f}")

    # Save results
    output = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "results": all_results,
        "overall_average": overall,
        "elapsed_seconds": round(elapsed, 1)
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to results.json")
    return output


if __name__ == "__main__":
    main()