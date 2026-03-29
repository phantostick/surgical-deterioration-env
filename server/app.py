"""
FastAPI server for the Surgical Deterioration Environment.
Exposes /health, /reset, /step, /state, /grade endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AgentAction, EpisodeState, GraderResult,
    ResetRequest, StepResult, WardObservation
)
from environment import SurgicalDeteriorationEnv

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Surgical Deterioration Early Warning Environment",
    description=(
        "An OpenEnv-compliant environment where an AI agent monitors a "
        "post-surgical ward and must detect patient deterioration early."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Single global environment instance (stateful per session)
env = SurgicalDeteriorationEnv()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "environment": "surgical_deterioration_env", "version": "1.0.0"}


@app.post("/reset", response_model=WardObservation)
def reset(request: ResetRequest = None):
    """
    Reset the environment and return the initial observation.
    - task_id: which of the 3 tasks to run
    - seed: random seed for reproducibility
    """
    if request is None:
        request = ResetRequest()
    try:
        obs = env.reset(request)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: AgentAction):
    """
    Take one action in the environment.
    - patient_id: which patient to act on
    - action: monitor | call_doctor | rapid_response | order_labs
    """
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EpisodeState)
def state():
    """
    Return current episode metadata without advancing the environment.
    """
    return env.state()


@app.post("/grade", response_model=GraderResult)
def grade():
    """
    Grade the current episode and return a score 0.0-1.0.
    Can be called at any time but most meaningful after episode ends.
    """
    try:
        result = env.grade()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    from environment import TASK_CONFIGS
    return {
        "tasks": [
            {
                "id": task_id,
                "description": cfg["description"],
                "n_patients": cfg["n_patients"],
                "max_steps": cfg["max_steps"],
                "difficulty": ["easy", "medium", "hard"][i]
            }
            for i, (task_id, cfg) in enumerate(TASK_CONFIGS.items())
        ]
    }

def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
