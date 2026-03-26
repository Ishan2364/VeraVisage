# Deepfake-Resistant Face Authentication

An academic project implementing a modular face-detection–based authentication
pipeline with multi-signal liveness detection to defend against deepfake attacks.

## Architecture
See the directory tree in the project report for a full breakdown of modules.

## Quick Start
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/enroll | Enrol a new identity |
| POST | /api/v1/authenticate | Authenticate a live face |
| GET  | /health | Service health check |
