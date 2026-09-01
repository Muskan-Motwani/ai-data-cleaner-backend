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
import multiprocessing as mp
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

MAX_SESSIONS = 5
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
MAX_ROWS = 5000


def _store_session(session_id: str, data: dict) -> None:
    SESSIONS[session_id] = data
    SESSIONS.move_to_end(session_id)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)


class SuggestRequest(BaseModel):
    session_id: str


class ApplyRequest(BaseModel):
    session_id: str
    steps: list[dict]


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

    original_row_count = len(df)
    was_sampled = False
    if original_row_count > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        was_sampled = True

    session_id = str(uuid.uuid4())
    _store_session(session_id, {"original": df, "cleaned": None})

    report = profile_and_detect_issues(df)

    return {
        "session_id": session_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "was_sampled": was_sampled,
        "original_row_count": original_row_count,
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


def _run_cleaning_in_subprocess(df: pd.DataFrame, steps: list[dict], result_queue: "mp.Queue"):
    """Runs in a separate process. If pandas/numpy hits a native crash here,
    only this subprocess dies — the main FastAPI server keeps running."""
    try:
        cleaned_df, applied_log = apply_cleaning_steps(df, steps)
        result_queue.put({"ok": True, "cleaned_df": cleaned_df, "log": applied_log})
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})


@app.post("/apply-cleaning")
async def apply_cleaning(req: ApplyRequest):
    """Applies the user-approved cleaning steps and returns a before/after summary.

    Runs the actual cleaning in a separate process. This is deliberate: if a
    native pandas/numpy operation crashes (segfault) instead of raising a normal
    Python exception, a plain try/except in this process cannot catch it — the
    whole server process would die with it, killing every other in-flight
    request and losing all sessions. Isolating the work in a subprocess means
    a crash there is contained: we detect it, return a clean error to this one
    request, and the main API server keeps running normally for everyone else.
    """
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Please upload the dataset again.")

    df = session["original"]

    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_run_cleaning_in_subprocess, args=(df, req.steps, result_queue))
    process.start()
    process.join(timeout=45)

    if process.is_alive():
        process.terminate()
        process.join()
        raise HTTPException(504, "Cleaning took too long and was stopped. Try with fewer steps or a smaller dataset.")

    if process.exitcode != 0:
        raise HTTPException(
            500,
            f"The cleaning process crashed unexpectedly (exit code {process.exitcode}). "
            "This usually means one of the selected steps hit an unstable operation on this data. "
            "Try unchecking a few steps (especially 'remove outliers') and applying again.",
        )

    if result_queue.empty():
        raise HTTPException(500, "Cleaning process ended with no result. Please try again.")

    result = result_queue.get()
    if not result["ok"]:
        raise HTTPException(400, f"Could not apply cleaning steps: {result['error']}")

    cleaned_df = result["cleaned_df"]
    applied_log = result["log"]

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
