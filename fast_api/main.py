"""
Step 1 — bare minimum FastAPI app.
Run with: uvicorn main:app --reload
Then open: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI

app = FastAPI(title="Face Pipeline API - Step 1")


@app.get("/")
def root():
    return {"message": "API is alive"}


@app.get("/hello")
def hello(name: str = "world"):
    return {"message": f"Hello, {name}!"}