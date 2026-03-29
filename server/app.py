"""
FastAPI server for the Surgical Deterioration Environment.
Exposes /health, /reset, /step, /state, /grade endpoints.
"""

import sys
import os
sys.path.insert(0, "/app")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <body style="font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px;">
        <h1>🏥 Surgical Deterioration Early Warning Environment</h1>
        <p>An OpenEnv-compliant environment for training AI agents to detect 
        post-surgical patient deterioration early.</p>
        <h2>Endpoints</h2>
        <ul>
            <li><a href="/health">GET /health</a> — Health check</li>
            <li><a href="/tasks">GET /tasks</a> — List all tasks</li>
            <li><a href="/state">GET /state</a> — Current episode state</li>
            <li>POST /reset — Reset environment</li>
            <li>POST /step — Take action</li>
            <li>POST /grade — Grade episode</li>
        </ul>
        <p><a href="/docs">📖 Interactive API Docs (Swagger)</a></p>
    </body>
    </html>
    """


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


# ---------------------------------------------------------------------------
# Required by OpenEnv spec
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()