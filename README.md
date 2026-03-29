# 🏥 Surgical Deterioration Early Warning Environment
title: Surgical Deterioration Early Warning Environment
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit

An OpenEnv-compliant environment where an AI agent acts as a clinical early-warning system in a post-surgical ward. The agent monitors patient vitals, detects deterioration before it becomes critical, and decides when to escalate care — balancing sensitivity against specificity.

**80% of preventable hospital deaths show warning signs hours before the event.** This environment trains and evaluates agents on exactly that window.

---

## Environment Description

The agent monitors a post-surgical ward with 1–20 patients depending on the task. Each patient has 10 vitals tracked every 30 minutes. Some patients are secretly deteriorating — the agent must identify them and escalate before they code.

### Key mechanics
- **NEWS2 scoring** — real clinical early warning formula used in NHS hospitals
- **Dense rewards** — every step produces a signal, not just episode end
- **False alarm penalty** — unnecessary rapid response activations reduce credibility score
- **Seeded simulation** — same seed + same actions = identical outcome every time
- **4 deterioration types** — sepsis, cardiac, respiratory, hemorrhage (different vital patterns)

---

## Action Space

| Action | When to use |
|--------|-------------|
| `monitor` | Patient is stable, NEWS2 < 3 |
| `call_doctor` | Concerning trend, NEWS2 3–6 |
| `rapid_response` | Critical deterioration, NEWS2 ≥ 7 |
| `order_labs` | Abnormal vitals, cause unclear |

---

## Observation Space

Per patient:
- `heart_rate`, `systolic_bp`, `diastolic_bp`, `respiratory_rate`
- `spo2`, `temperature`, `urine_output_ml_hr`, `gcs`, `pain_score`
- `news2_score` (computed from above)
- `nursing_flags` (free-text clinical observations)

Ward level:
- `rapid_response_available`, `false_alarm_count`, `credibility_score`
- `hours_remaining`, `step`

---

## Tasks

| Task | Patients | Steps | Difficulty |
|------|----------|-------|------------|
| `task1_single_patient_escalation` | 1 | 8 | Easy |
| `task2_subtle_deterioration` | 5 | 24 | Medium |
| `task3_ward_triage` | 20 | 24 | Hard |

### Task 1 — Single Patient Escalation (Easy)
One patient, clearly deteriorating. Does the agent escalate? How early?

### Task 2 — Subtle Deterioration Detection (Medium)
5 patients, one deteriorating slowly across multiple vitals. Identify the correct patient and escalate before a critical event. False alarms on stable patients penalize score.

### Task 3 — Full Ward Triage Under Scarcity (Hard)
20 patients, 3 deteriorating at different speeds. Rapid response limited to once every 2 steps. Each patient that codes = -0.2 penalty. Requires genuine clinical reasoning to score above 0.5.

---

## Reward Function

```
Step reward:
  +0.8  correct rapid response activation
  +0.3  correct call_doctor
  +0.1  order_labs on deteriorating patient
  +0.02 monitor on stable patient
  -0.3  false rapid response (stable patient)
  -1.0  patient codes (missed deterioration)

Episode score (0.0–1.0):
  Task 1: escalation timing + false alarm penalty
  Task 2: correct patient identified + early detection + false alarms
  Task 3: 0.4×lives_saved + 0.3×(1-false_alarm_rate) + 0.3×early_detection - 0.2×coded
```

---

## Setup

### Local

```bash
# Clone repo
git clone <your-repo-url>
cd surgical_deterioration_env

# Install dependencies
pip install -r server/requirements.txt

# Start environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run baseline
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token"
python inference.py
```

### Docker

```bash
docker build -t surgical-deterioration-env .
docker run -p 8000:8000 surgical-deterioration-env
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Reset environment, returns initial observation |
| POST | `/step` | Take action, returns observation + reward |
| GET | `/state` | Current episode metadata |
| POST | `/grade` | Score current episode |
| GET | `/tasks` | List all tasks |

### Example

```python
import requests

# Reset
obs = requests.post("http://localhost:8000/reset", json={
    "task_id": "task1_single_patient_escalation",
    "seed": 42
}).json()

# Step
result = requests.post("http://localhost:8000/step", json={
    "patient_id": 0,
    "action": "call_doctor"
}).json()

# Grade
score = requests.post("http://localhost:8000/grade").json()
print(score["score"])  # 0.0–1.0
```

---

## Baseline Scores

| Task | Average Score | Notes |
|------|--------------|-------|
| task1 | ~0.70 | LLM handles obvious cases well |
| task2 | ~0.40 | Subtle detection is challenging |
| task3 | ~0.25 | Ward triage requires strategic reasoning |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API key |
| `ENV_BASE_URL` | Environment server URL (default: http://localhost:8000) |
