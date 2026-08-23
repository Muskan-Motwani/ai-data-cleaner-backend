# AI Data Cleaner — Backend

Companion tool to **VizPilot (AI Dashboard Maker)**. Takes a raw/messy dataset,
detects concrete data-quality issues, and uses Gemini to suggest an
executable cleaning + feature-engineering plan — so a dataset is
dashboard-ready before it ever reaches VizPilot.

## How it works

1. **`POST /upload`** — accepts CSV/XLSX, profiles it, and detects concrete
   issues: missing values (with severity), duplicate rows, outliers (IQR
   method), numeric columns stored as text, high-cardinality columns.
2. **`POST /suggest-cleaning`** — sends the issue report to Gemini, gets back
   a structured cleaning plan (specific steps like "fill missing values in
   'age' with median") plus 2-4 feature engineering ideas.
3. **`POST /apply-cleaning`** — takes the user-approved steps, actually runs
   them with pandas, returns a before/after comparison.
4. **`GET /download/{session_id}`** — downloads the cleaned dataset as CSV.

## Why this exists

VizPilot (the dashboard generator) works best on clean data — but real-world
datasets are messy. This tool is the upstream step: clean first, then hand
off to VizPilot for dashboard generation. Together they form a full
"raw data → clean data → dashboard" pipeline.

## Tech stack

- **Backend:** Python, FastAPI, pandas
- **AI:** Google Gemini API (`gemini-2.5-flash`), function calling for
  schema-validated cleaning plans
- **Deployment:** Render (same platform as VizPilot backend — Railway was
  tried first but its IP range gets blocked by some Indian ISPs, so this
  project skips straight to Render)

## Key engineering decisions

- **Issue detection is rule-based, not AI-based** — missing %, duplicates,
  and outliers (IQR method) are computed directly with pandas. Only the
  *plan* (what to do about each issue) comes from the LLM. This is
  deliberate: deterministic detection is more reliable and cheaper than
  asking an LLM to "notice" problems in raw stats.
- **Steps are structured actions, not free-text instructions** — each
  cleaning step is a small JSON object (`{"action": "fill_missing",
  "column": "age", "strategy": "median"}`) that maps directly to a pandas
  operation. This means the AI's plan is always executable, never
  ambiguous.
- **Unknown actions fail soft, not hard** — if the AI suggests an
  unsupported action, that one step is skipped and logged rather than
  crashing the whole cleaning run.
- **Reused proven patterns from VizPilot** — encoding-safe CSV reading
  (`utf-8 → latin-1 → cp1252` fallback) and NaN-safe JSON serialization were
  carried over directly, since both were real bugs hit and fixed in the
  first project.

## Setup

```bash
git clone <this-repo>
cd data-cleaner-backend
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
uvicorn main:app --reload
```

Full API docs at `/docs` once running.

## Deploying (Render, free tier)

1. Push this folder to a GitHub repo
2. On render.com: New → Web Service → connect the repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GEMINI_API_KEY`
6. Deploy — note the free tier sleeps after 15 min of inactivity, so the
   first request after idle time takes ~30-50s to wake up

## Known limitations

- In-memory session storage (fine for a demo; would move to Redis/S3 for
  production)
- Free-tier hosting has a 512MB RAM ceiling — very large files may fail;
  works reliably on typical portfolio-sized datasets (tens of thousands of
  rows)
- Feature engineering is currently limited to date extraction and simple
  ratios — a natural next step would be AI-suggested binning, one-hot
  encoding, or text feature extraction
