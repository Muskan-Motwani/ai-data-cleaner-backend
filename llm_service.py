"""
Talks to Google Gemini (free tier) to turn a detected-issues report into a
list of concrete, executable cleaning steps + feature engineering ideas.
Uses function calling so we get reliable, schema-validated JSON — same
pattern proven in the VizPilot dashboard-maker backend.

Get a free API key (no card needed): https://aistudio.google.com/apikey
Set it as the GEMINI_API_KEY environment variable.
"""

import os
import json
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-2.5-flash"

CLEANING_TOOL = {
    "name": "generate_cleaning_plan",
    "description": "Return a concrete, executable data cleaning and feature engineering plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "1-2 sentence overview of the dataset's main quality issues",
            },
            "cleaning_steps": {
                "type": "array",
                "description": "Concrete cleaning actions to apply, in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "drop_duplicates",
                                "drop_column",
                                "fill_missing",
                                "convert_dtype",
                                "remove_outliers",
                                "strip_whitespace",
                                "standardize_case",
                            ],
                        },
                        "column": {"type": "string", "description": "target column (omit for drop_duplicates)"},
                        "strategy": {
                            "type": "string",
                            "enum": ["mean", "median", "mode", "zero", "unknown", "drop_rows"],
                            "description": "only used when action is fill_missing",
                        },
                        "target_type": {
                            "type": "string",
                            "enum": ["numeric", "datetime", "category"],
                            "description": "only used when action is convert_dtype",
                        },
                        "case": {
                            "type": "string",
                            "enum": ["lower", "upper", "title"],
                            "description": "only used when action is standardize_case",
                        },
                        "reasoning": {"type": "string", "description": "1 sentence why this step is needed"},
                    },
                    "required": ["action", "reasoning"],
                },
            },
            "feature_engineering_ideas": {
                "type": "array",
                "description": "2-4 suggested new features that would help downstream analysis/dashboards",
                "items": {
                    "type": "object",
                    "properties": {
                        "new_column_name": {"type": "string"},
                        "method": {
                            "type": "string",
                            "enum": ["extract_year", "extract_month", "ratio"],
                        },
                        "source_column": {"type": "string", "description": "used for extract_year/extract_month"},
                        "column_a": {"type": "string", "description": "used for ratio"},
                        "column_b": {"type": "string", "description": "used for ratio"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["new_column_name", "method", "reasoning"],
                },
            },
        },
        "required": ["summary", "cleaning_steps", "feature_engineering_ideas"],
    },
}


def get_cleaning_suggestions(issue_report: dict) -> dict:
    prompt = f"""You are a senior data analyst preparing a raw dataset for dashboarding.

Here is a profiling + issue-detection report for the dataset:
{json.dumps(issue_report, indent=2)}

Based on the detected issues, propose a concrete cleaning plan:
- For every issue in the "issues" list, suggest a specific, executable cleaning step
- Only reference column names that actually exist in the report above
- Also suggest 2-4 feature engineering ideas that would make this dataset more useful
  for building an analytics dashboard afterward (e.g. extracting year/month from a
  date column, or creating a ratio between two related numeric columns)
- Keep cleaning steps minimal and justified — don't suggest dropping columns or rows
  unless the issue report shows a real problem (e.g. >30% missing, near-duplicate)

Call the generate_cleaning_plan function with your plan."""

    model = genai.GenerativeModel(
        model_name=MODEL,
        tools=[{"function_declarations": [CLEANING_TOOL]}],
    )

    response = model.generate_content(
        prompt,
        tool_config={"function_calling_config": {"mode": "ANY"}},
    )

    for part in response.candidates[0].content.parts:
        if part.function_call:
            return _to_plain(part.function_call.args)

    return {"error": "No suggestion generated"}


def _to_plain(obj):
    """Recursively converts Gemini's protobuf-backed args into plain
    Python dict/list/str/number so they're JSON-serializable."""
    if hasattr(obj, "items"):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (str, bytes)):
        return obj
    if hasattr(obj, "__iter__"):
        return [_to_plain(v) for v in obj]
    return obj
