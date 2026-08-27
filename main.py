"""
AI Data Cleaner - Backend
--------------------------
Companion tool to VizPilot (AI Dashboard Maker). Takes a messy dataset and
walks it through: issue detection -> AI-suggested cleaning steps -> apply
cleaning -> download cleaned file.

Flow:
1. POST /upload            -> profiles dataset + detects issues (missing %,
                               duplicates, outliers, dtype problems)
2. POST /suggest-cleaning   -> asks Gemini for cleaning steps + feature
                               engineering ideas (structured JSON)
3. POST /apply-cleaning     -> applies chosen steps, returns before/after summary
4. GET  /download/{session_id} -> downloads the cleaned CSV
"""

import os
import io
import uuid
from collections import OrderedDict

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from analyzer import profile_and_detect_issues
from cleaner import apply_cleaning_steps
from llm_service import get_cleaning_suggestions

app = FastAPI(title="AI Data Cleaner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> {"original": df, "cleaned": df or None}
# Capped + LRU-evicted: free-tier hosting has ~512MB RAM, and every retry
# (including the frontend's automatic re-upload-on-failure) creates a new
# session holding a full dataframe. Without a cap, repeated retries or
# uploads can silently accumulate multiple large datasets in memory until
# the process is OOM-killed. Keeping only the most recent few sessions
# keeps memory bounded regardless of how many times a user retries.
MAX_SESSIONS = 5
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()


def _store_session(session_id: str, data: dict) -> None:
    SESSIONS[session_id] = data
    SESSIONS.move_to_end(session_id)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)  # evict the oldest session


class SuggestRequest(BaseModel):
    session_id: str


class ApplyRequest(BaseModel):
    session_id: str
    steps: list[dict]  # the cleaning_steps array the user approved (from /suggest-cleaning)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Data Cleaner backend is running"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Accepts a CSV/XLSX, profiles it, and detects data quality issues."""
    if not file.filename.endswith((".csv", ".xlsx")):
        raise HTTPException(400, "Please upload a .csv or .xlsx file")

    contents = await file.read()

    try:
        if file.filename.endswith(".csv"):
            df = None
            last_error = None
            for encoding in ("utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError) as e:
                    last_error = e
                    continue
            if df is None:
                raise HTTPException(400, f"Could not decode CSV file: {last_error}")
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")

    session_id = str(uuid.uuid4())
    _store_session(session_id, {"original": df, "cleaned": None})

    report = profile_and_detect_issues(df)

    return {
        "session_id": session_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "report": report,
    }


@app.post("/suggest-cleaning")
async def suggest_cleaning(req: SuggestRequest):
    """Sends the issue report to Gemini and gets back suggested cleaning steps
    + feature engineering ideas."""
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Please upload the dataset again.")

    report = profile_and_detect_issues(session["original"])
    suggestions = get_cleaning_suggestions(report)
    return suggestions


@app.post("/apply-cleaning")
async def apply_cleaning(req: ApplyRequest):
    """Applies the user-approved cleaning steps and returns a before/after summary."""
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Please upload the dataset again.")

    df = session["original"]
    try:
        cleaned_df, applied_log = apply_cleaning_steps(df, req.steps)
    except Exception as e:
        raise HTTPException(400, f"Could not apply cleaning steps: {e}")

    session["cleaned"] = cleaned_df
    SESSIONS.move_to_end(req.session_id)

    after_report = profile_and_detect_issues(cleaned_df)

    return {
        "rows_before": len(df),
        "rows_after": len(cleaned_df),
        "columns_before": len(df.columns),
        "columns_after": len(cleaned_df.columns),
        "applied_steps": applied_log,
        "after_report": after_report,
    }


@app.get("/download/{session_id}")
async def download_cleaned(session_id: str):
    """Streams the cleaned CSV back to the user."""
    session = SESSIONS.get(session_id)
    if session is None or session["cleaned"] is None:
        raise HTTPException(404, "No cleaned dataset found for this session. Run /apply-cleaning first.")

    buffer = io.StringIO()
    session["cleaned"].to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_dataset.csv"},
    )
