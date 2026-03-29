"""
FastAPI server for the Surgical Deterioration Environment.
Exposes /health, /reset, /step, /state, /grade endpoints.
"""

import sys
import os
import time
from uuid import uuid4
sys.path.insert(0, "/app")

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

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

# Session-based environment store: session_id -> (env, timestamp)
envs: dict = {}
MAX_SESSION_AGE = 3600  # 1 hour


def cleanup_old_sessions():
    now = time.time()
    dead = [k for k, (_, ts) in envs.items() if now - ts > MAX_SESSION_AGE]
    for k in dead:
        del envs[k]


def get_env(session_id: str) -> SurgicalDeteriorationEnv:
    if session_id not in envs:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    env, _ = envs[session_id]
    envs[session_id] = (env, time.time())  # refresh timestamp
    return env


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
            <li>POST /reset — Reset environment, returns session_id</li>
            <li>POST /step?session_id=... — Take action</li>
            <li>GET /state?session_id=... — Current episode state</li>
            <li>POST /grade?session_id=... — Grade episode</li>
        </ul>
        <p><a href="/docs">📖 Interactive API Docs (Swagger)</a></p>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "environment": "surgical_deterioration_env", "version": "1.0.0"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Reset the environment and return the initial observation + session_id.
    Use session_id in all subsequent calls.
    """
    cleanup_old_sessions()
    if request is None:
        request = ResetRequest()
    try:
        session_id = str(uuid4())
        env = SurgicalDeteriorationEnv()
        obs = env.reset(request)
        envs[session_id] = (env, time.time())
        result = obs.dict()
        result["session_id"] = session_id
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: AgentAction, session_id: str = "default"):
    """
    Take one action in the environment.
    Pass session_id from /reset as query param: /step?session_id=...
    """
    # Support default single-session for backward compatibility
    if session_id == "default":
        if "default" not in envs:
            env = SurgicalDeteriorationEnv()
            envs["default"] = (env, time.time())
        env = get_env("default")
    else:
        env = get_env(session_id)
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EpisodeState)
def state(session_id: str = "default"):
    """Return current episode metadata."""
    if session_id == "default" and "default" not in envs:
        env = SurgicalDeteriorationEnv()
        envs["default"] = (env, time.time())
    env = get_env(session_id)
    return env.state()


@app.post("/grade", response_model=GraderResult)
def grade(session_id: str = "default"):
    """Grade the current episode and return a score 0.0-1.0."""
    if session_id == "default" and "default" not in envs:
        env = SurgicalDeteriorationEnv()
        envs["default"] = (env, time.time())
    env = get_env(session_id)
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