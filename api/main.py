"""
main.py
───────
FastAPI application factory: registers all routers, middleware, and startup events.
"""
from fastapi import FastAPI
from api.routers import enroll, authenticate

app = FastAPI(
    title="Deepfake-Resistant Face Authentication API",
    description="Liveness-checked face verification service.",
    version="0.1.0",
)

app.include_router(enroll.router, prefix="/api/v1")
app.include_router(authenticate.router, prefix="/api/v1")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}
