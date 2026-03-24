# Databricks notebook source
# MAGIC %md
# MAGIC # ⚡ EV Platform — Automated Documentation Generator
# MAGIC **Version:** 2.0.0
# MAGIC
# MAGIC Generates a **single, unified, RAG-optimised** markdown document per source system.
# MAGIC The document is structured to serve Technical Engineers, Support Teams,
# MAGIC Business/Leadership, and Newcomers — all from one file.
# MAGIC
# MAGIC **RAG Design Principles applied:**
# MAGIC - Every entity (table, column, function) named explicitly in every section — no pronouns
# MAGIC - Q&A pairs embedded throughout — chatbot pre-indexes real questions + answers
# MAGIC - Self-contained sections — each chunk makes sense in isolation after retrieval
# MAGIC - Dense, descriptive section headers for semantic routing
# MAGIC - Failure modes as first-class content — most chatbot queries are debugging queries
# MAGIC - Full glossary — chatbot can answer "what is EUH?" without needing code context

# COMMAND ----------
# MAGIC %md ## Cell 1 — Install Dependencies

# COMMAND ----------

# %pip install requests
# dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## Cell 2 — Widgets

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("source_name", "", "Source Name (e.g. driivz)")
dbutils.widgets.dropdown("force_regenerate", "false", ["true", "false"], "Force Regenerate")
dbutils.widgets.dropdown("log_level", "INFO", ["DEBUG", "INFO", "WARNING"], "Log Level")

source          = dbutils.widgets.get("source_name").lower().strip()
force_regenerate = dbutils.widgets.get("force_regenerate") == "true"
log_level       = dbutils.widgets.get("log_level")

if not source:
    raise ValueError("❌ 'source_name' widget is required. Example: 'driivz'")

print(f"✅ Source        : {source}")
print(f"✅ Force Regen   : {force_regenerate}")
print(f"✅ Log Level     : {log_level}")

# COMMAND ----------
# MAGIC %md ## Cell 3 — Config & Imports

# COMMAND ----------

import base64
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import requests
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, TimestampType

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("doc_generator")

# ── Platform Config ───────────────────────────────────────────────────────────
WORKSPACE_URL = "https://<your-workspace-url>"
TOKEN         = dbutils.secrets.get(scope="platform", key="workspace_pat")
HEADERS       = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# ── Storage Config ────────────────────────────────────────────────────────────
DOC_OUTPUT_VOLUME = "/Volumes/platform/docs/generated"
STATE_TABLE       = "platform.engineering.workflow_doc_state"
NOTEBOOK_ROOT     = "/Workspace"

# ── LLM Config ───────────────────────────────────────────────────────────────
# Single model — Llama 3.3 70B for both stages. Most cost-effective stable
# pay-per-token model on Databricks as of March 2026.
#
# Retired/unavailable: 405B (Feb 2026), Claude 3.5 Sonnet (unavailable),
#   Claude 3.7 Sonnet (retiring Apr 2026), Llama 4 Maverick (Mar 2026).
#
# Stage 1 quality issues were prompt + token problems, not model problems.
# Stage 2 uses TWO sequential calls (technical + operational halves) to
# stay within 70B output limits and avoid trailing-off on long outputs.
#
# Alternative if available in your workspace:
#   LLM_ENDPOINT = "databricks-gemini-3-1-flash-lite"  # slightly stronger on long structured output

LLM_ENDPOINT          = "databricks-meta-llama-3-3-70b-instruct"
LLM_STAGE1_MAX_TOKENS = 3000    # per-notebook summary (or per-half for large notebooks)
LLM_STAGE2_MAX_TOKENS = 5000    # per Stage 2 half-doc (called twice)

# Shared
LLM_TEMPERATURE   = 0.1
LLM_RETRY_LIMIT   = 3

# Timeouts — 70B on large prompts routinely takes 3-5 minutes.
# Retry wait must also be long: retrying immediately after a 5-min timeout is pointless.
LLM_TIMEOUT_STAGE1 = 300       # seconds — per-notebook summary call
LLM_TIMEOUT_STAGE2 = 480       # seconds — unified doc half (bigger prompt + output)
LLM_RETRY_WAIT     = 30        # seconds — base wait after timeout (doubles each retry)

# ── Processing Config ─────────────────────────────────────────────────────────
# MAX_CODE_CHARS: soft limit per LLM call. Notebooks ABOVE this are split into
# two halves, each summarised separately, then merged. This keeps any single
# LLM input manageable and prevents timeouts on very large notebooks.
MAX_CODE_CHARS     = 25_000    # chars per LLM call (~6K tokens input — safe for 70B)
MAX_PARALLEL_CALLS = 2         # reduced from 4 — avoids saturating the endpoint

log.info("Config loaded for source: %s", source)

# COMMAND ----------
# MAGIC %md ## Cell 4 — Workspace API Helpers

# COMMAND ----------

def list_notebooks(path: str = NOTEBOOK_ROOT) -> list[str]:
    """
    Recursively list all notebook paths under `path`.
    Skips inaccessible directories silently (404).
    """
    url = f"{WORKSPACE_URL}/api/2.0/workspace/list"
    notebooks = []

    try:
        r = requests.get(url, headers=HEADERS, params={"path": path}, timeout=30)
        r.raise_for_status()
    except requests.HTTPError:
        if r.status_code == 404:
            log.debug("Skipping inaccessible path: %s", path)
            return []
        raise

    for obj in r.json().get("objects", []):
        if obj.get("object_type") == "NOTEBOOK":
            notebooks.append(obj["path"])
        elif obj.get("object_type") == "DIRECTORY":
            notebooks.extend(list_notebooks(obj["path"]))

    return notebooks


def export_notebook(path: str) -> Optional[str]:
    """
    Export notebook as SOURCE format.
    IMPORTANT: Workspace Export API returns base64-encoded content — must decode.
    Returns None on failure (non-fatal).
    """
    url    = f"{WORKSPACE_URL}/api/2.0/workspace/export"
    params = {"path": path, "format": "SOURCE"}

    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        encoded = r.json().get("content", "")
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception as e:
        log.warning("Failed to export notebook %s: %s", path, e)
        return None

# COMMAND ----------
# MAGIC %md ## Cell 5 — Source Filtering

# COMMAND ----------

def belongs_to_source(path: str, source: str) -> bool:
    """
    3-tier matching: directory segment → filename prefix → loose substring.
    Intentionally broad — false positives safer than missed notebooks.
    """
    name  = path.split("/")[-1].lower()
    parts = [p.lower() for p in path.split("/")]
    s     = source.lower()

    if s in parts[:-1]:
        return True

    prefixes = [
        f"landing_etl_{s}", f"raw_etl_{s}", f"euh_etl_{s}", f"curated_etl_{s}",
        f"{s}_landing", f"{s}_raw", f"{s}_euh", f"{s}_curated",
        f"{s}_ingestion", f"{s}_transform", f"{s}_",
    ]
    if any(name.startswith(p) or name == p for p in prefixes):
        return True

    if s in name:
        return True

    return False


all_paths      = list_notebooks(NOTEBOOK_ROOT)
workflow_paths = [p for p in all_paths if belongs_to_source(p, source)]

if not workflow_paths:
    raise Exception(f"❌ No notebooks found for source '{source}'.")

log.info("Matched %d notebooks for source '%s':", len(workflow_paths), source)
for p in workflow_paths:
    print(" ✓", p)

# COMMAND ----------
# MAGIC %md ## Cell 6 — LLM Call with Retry

# COMMAND ----------

def call_llm(
    prompt: str,
    context: str = "",
    max_tokens: int = LLM_STAGE1_MAX_TOKENS,
    timeout: int = LLM_TIMEOUT_STAGE1,
) -> str:
    """
    Call Databricks model serving with exponential backoff retry.

    Args:
        prompt:     The user prompt.
        context:    Label for logging (e.g. notebook name).
        max_tokens: Output token limit.
        timeout:    HTTP read timeout in seconds. Use LLM_TIMEOUT_STAGE1 (300s)
                    for notebook summaries and LLM_TIMEOUT_STAGE2 (480s) for
                    unified doc calls. Do NOT use 120s — 70B on large prompts
                    regularly takes 3-5 minutes to respond.

    Retry strategy:
        - HTTP 4xx (except 429): non-retryable, raise immediately.
        - HTTP 5xx, 429, or timeout: retry up to LLM_RETRY_LIMIT times.
        - Wait = LLM_RETRY_WAIT * 2^(attempt-1). Starts at 30s because
          retrying immediately after a 5-minute timeout is pointless.
    """
    url     = f"{WORKSPACE_URL}/serving-endpoints/{LLM_ENDPOINT}/invocations"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    log.debug("LLM → max_tokens=%d  timeout=%ds  context=%s", max_tokens, timeout, context)

    last_error = None
    for attempt in range(1, LLM_RETRY_LIMIT + 1):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        except requests.HTTPError as e:
            if r.status_code < 500 and r.status_code != 429:
                raise                       # 4xx errors are not transient — fail fast
            last_error = e
            wait = LLM_RETRY_WAIT * (2 ** (attempt - 1))
            log.warning("LLM attempt %d/%d HTTP error: %s — retrying in %ds",
                        attempt, LLM_RETRY_LIMIT, e, wait)
            time.sleep(wait)

        except requests.Timeout as e:
            last_error = e
            wait = LLM_RETRY_WAIT * (2 ** (attempt - 1))
            log.warning("LLM attempt %d/%d TIMEOUT after %ds — retrying in %ds",
                        attempt, LLM_RETRY_LIMIT, timeout, wait)
            time.sleep(wait)

        except Exception as e:
            last_error = e
            wait = LLM_RETRY_WAIT * (2 ** (attempt - 1))
            log.warning("LLM attempt %d/%d error: %s — retrying in %ds",
                        attempt, LLM_RETRY_LIMIT, e, wait)
            time.sleep(wait)

    raise RuntimeError(
        f"LLM failed after {LLM_RETRY_LIMIT} attempts [{context}]: {last_error}"
    )

# COMMAND ----------
# MAGIC %md ## Cell 7 — Stage 1: Per-Notebook Deep Summary

# COMMAND ----------

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN NOTE:
# Stage 1 extracts deep, structured facts from each notebook.
# These facts become the raw material for Stage 2.
# We extract MORE than we need here — it's cheaper to filter than to re-call.
# ─────────────────────────────────────────────────────────────────────────────

NOTEBOOK_SUMMARY_PROMPT = """\
You are a senior data engineer performing a deep technical analysis of a Databricks notebook.

Source System : {source}
Notebook Name : {name}
Notebook Path : {path}

Your output will feed a RAG knowledge base that powers an engineering chatbot.
COMPLETENESS IS CRITICAL — missing a transformation or column means the chatbot
cannot answer questions about it. Do not summarise, skip, or generalise.

Always use exact names from the code: table names, column names, function names,
variable names, hardcoded values. Never say "the column" — always say the column name.

---

**1. NOTEBOOK PURPOSE**
One precise sentence. Name the notebook, name what it does, name the source system.

**2. DATA LAYER**
Which layer? (Landing / Raw / EUH / Curated). Why?

**3. INPUTS**
For EVERY data source read, list:
- Exact table/file/API name (as it appears in the code)
- Format (Delta, Parquet, REST API, etc.)
- Every column referenced from this source (list them all)

**4. OUTPUTS**
For EVERY table or file written, list:
- Exact table/file name (as it appears in the code)
- Format and write mode (overwrite / append / merge / upsert)
- Every column written (list them all with data types if castable from code)

**5. COLUMN-LEVEL TRANSFORMATION LOGIC**
This is the most important section. For EVERY column that is created, renamed,
cast, enriched, derived, or conditionally set — document it as:

  Column: <exact_column_name>
  Source: <where it comes from — input column name, expression, or constant>
  Logic: <exact transformation applied — cast, concat, coalesce, condition, formula>
  Business meaning: <what this column represents in business terms>

Do NOT group columns together. Every column gets its own entry.
Do NOT skip columns that "seem obvious". Include all of them.

**6. TRANSFORMATION STEPS (execution order)**
List every transformation step in the order it runs:
  Step N: <function or operation name>
  Input: <input dataframe/table and key columns>
  Operation: <exactly what happens — filter condition, join key, dedup key, window spec>
  Output: <resulting dataframe/table and what changed>
  Business reason: <why this step exists>

**7. BUSINESS RULES & HARDCODED VALUES**
For EVERY filter, condition, threshold, or hardcoded value in the code:
- Exact column name and condition (e.g. `status != 'CANCELLED'`)
- What this rule excludes or includes
- Business reason (if inferrable from context)

**8. DEDUPLICATION LOGIC**
If any deduplication occurs:
- Exact deduplication key columns (list all)
- Method used (dropDuplicates, window + row_number, merge, etc.)
- Which records are kept (first, last, highest value, etc.)
- What happens to the dropped duplicates

**9. JOIN LOGIC**
For EVERY join in the code:
- Left table (exact name)
- Right table (exact name)
- Join key column(s) (exact names on both sides)
- Join type (inner, left, right, cross)
- Columns brought in from the right table
- Business reason for this join

**10. ERROR HANDLING & DATA QUALITY**
- How are nulls handled? Which columns? What is the fallback value?
- Are there explicit null checks, assertions, or guards?
- What happens if input data is empty?
- What happens if a join produces no matches?

**11. DEPENDENCIES**
- Other notebooks called (exact paths)
- Shared utilities or functions imported (exact module names)
- External APIs or services called
- Tables that must exist before this notebook runs

**12. FAILURE MODES**
For EVERY realistic failure scenario:
- Failure name
- Root cause (exact condition that triggers it)
- Symptom / error message
- How to detect it
- Step-by-step fix

**13. PERFORMANCE NOTES**
- Any caching, broadcasting, repartitioning, or Z-ordering
- Approximate data volume (if inferrable)
- Potential bottlenecks

IMPORTANT: If the code is cut off or truncated, note exactly where it stops and
what sections you could not analyse. Do not fabricate content for unseen code.

Code:
{code}
"""


def clean_code(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# MAGIC"):
            continue
        if stripped.startswith("%") and not stripped.startswith("%sql"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _call_stage1(prompt: str, context: str) -> str:
    """Wrapper: Stage 1 always uses Stage 1 timeout."""
    return call_llm(
        prompt,
        context=context,
        max_tokens=LLM_STAGE1_MAX_TOKENS,
        timeout=LLM_TIMEOUT_STAGE1,
    )


def summarize_notebook(path: str, raw_content: str) -> dict:
    """
    Summarise a single notebook.

    Large notebook strategy (> MAX_CODE_CHARS):
    Instead of truncating, split the code into two halves and run two
    separate LLM calls. Each half gets the full prompt context (name, path,
    source) so the model knows what it is analysing. The two partial summaries
    are then merged with a lightweight third call.

    This avoids both truncation (missing transformation logic) AND timeouts
    (sending too much code in one call).
    """
    name    = path.split("/")[-1]
    cleaned = clean_code(raw_content)
    total   = len(cleaned)

    if total <= MAX_CODE_CHARS:
        # ── Small notebook: single call ───────────────────────────────────────
        log.info("Summarising (single call, %d chars): %s", total, name)
        prompt = NOTEBOOK_SUMMARY_PROMPT.format(
            source=source.title(), name=name, path=path, code=cleaned,
        )
        try:
            summary = _call_stage1(prompt, context=name)
            log.info("✓ Summarised: %s", name)
            return {"path": path, "name": name, "summary": summary, "error": None}
        except Exception as e:
            log.error("✗ Failed: %s — %s", name, e)
            return {"path": path, "name": name,
                    "summary": f"_Summary unavailable: {e}_", "error": str(e)}

    else:
        # ── Large notebook: split into two halves, then merge ─────────────────
        mid = total // 2
        # Split at a newline boundary near the midpoint to avoid cutting mid-line
        split_at = cleaned.rfind("\n", 0, mid) + 1
        if split_at <= 0:
            split_at = mid
        half_a = cleaned[:split_at]
        half_b = cleaned[split_at:]

        log.warning(
            "Large notebook '%s' (%d chars) — splitting into two halves "
            "(%d + %d chars) to avoid timeout.",
            name, total, len(half_a), len(half_b),
        )

        half_note_a = (
            "NOTE: This is the FIRST HALF of the notebook. "
            "Analyse only what is visible. A second half exists and will be merged."
        )
        half_note_b = (
            "NOTE: This is the SECOND HALF of the notebook. "
            "Analyse only what is visible. A first half was already analysed."
        )

        errors = []
        summary_a = summary_b = None

        try:
            prompt_a = NOTEBOOK_SUMMARY_PROMPT.format(
                source=source.title(), name=f"{name} (half 1/2)",
                path=path, code=half_a + f"\n\n# {half_note_a}",
            )
            summary_a = _call_stage1(prompt_a, context=f"{name}_half1")
            log.info("✓ Half 1/2 done: %s", name)
        except Exception as e:
            errors.append(f"half1: {e}")
            log.error("✗ Half 1/2 failed: %s — %s", name, e)

        try:
            prompt_b = NOTEBOOK_SUMMARY_PROMPT.format(
                source=source.title(), name=f"{name} (half 2/2)",
                path=path, code=half_b + f"\n\n# {half_note_b}",
            )
            summary_b = _call_stage1(prompt_b, context=f"{name}_half2")
            log.info("✓ Half 2/2 done: %s", name)
        except Exception as e:
            errors.append(f"half2: {e}")
            log.error("✗ Half 2/2 failed: %s — %s", name, e)

        if not summary_a and not summary_b:
            return {"path": path, "name": name,
                    "summary": f"_Both halves failed: {errors}_", "error": str(errors)}

        # If only one half succeeded, use it with a warning
        if not summary_a or not summary_b:
            partial = summary_a or summary_b
            warn = "_WARNING: Only one half of this notebook was successfully summarised._\n\n"
            return {"path": path, "name": name, "summary": warn + partial, "error": str(errors)}

        # ── Merge the two partial summaries ───────────────────────────────────
        merge_prompt = f"""You are a senior data engineer.
Two partial summaries of the same Databricks notebook have been generated separately.
Merge them into ONE complete, coherent summary following the same 13-section structure.

Rules:
- Combine information from both halves — do not drop anything.
- Deduplicate repeated points.
- Use exact names from the code (tables, columns, functions).
- If the two halves describe different transformation steps, keep all steps in execution order.

Notebook: {name}
Source: {source.title()}
Path: {path}

--- FIRST HALF SUMMARY ---
{summary_a}

--- SECOND HALF SUMMARY ---
{summary_b}

Produce the merged summary now:"""

        try:
            merged = _call_stage1(merge_prompt, context=f"{name}_merge")
            log.info("✓ Merged halves: %s", name)
            return {"path": path, "name": name, "summary": merged, "error": None}
        except Exception as e:
            log.error("✗ Merge failed: %s — falling back to concatenation: %s", name, e)
            # Fallback: just concatenate both summaries with a divider
            combined = (
                f"_NOTE: Merge step failed. Showing both halves concatenated._\n\n"
                f"### Half 1\n{summary_a}\n\n### Half 2\n{summary_b}"
            )
            return {"path": path, "name": name, "summary": combined, "error": str(e)}


def summarize_all_notebooks(paths: list, contents: list) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as executor:
        futures = {
            executor.submit(summarize_notebook, p, c): p
            for p, c in zip(paths, contents) if c is not None
        }
        for future in as_completed(futures):
            results.append(future.result())

    order = {p: i for i, p in enumerate(paths)}
    results.sort(key=lambda x: order.get(x["path"], 999))
    return results

# COMMAND ----------
# MAGIC %md ## Cell 8 — Stage 2: Single Unified RAG-Optimised Document

# COMMAND ----------

# ─────────────────────────────────────────────────────────────────────────────
# RAG DOCUMENT DESIGN — WHY THIS STRUCTURE:
#
# 1. METADATA HEADER — gives the chatbot source/owner/version context
#    for every chunk retrieved from this doc.
#
# 2. QUICK REFERENCE — dense facts (tables, owners, SLA) that answer
#    lookup questions instantly without needing full context.
#
# 3. BUSINESS OVERVIEW — written for non-technical readers.
#    Answers: "what does this pipeline do?" "what breaks if it fails?"
#
# 4. ARCHITECTURE & DATA FLOW — visual-style ASCII + explanation.
#    Answers: "how does data flow from source to curated?"
#
# 5. NOTEBOOK-BY-NOTEBOOK BREAKDOWN — deep technical per notebook.
#    Answers: "what does notebook X do?" "where is Y logic implemented?"
#
# 6. COMPLETE TABLE CATALOGUE — every table, every column.
#    Answers: "what columns does table X have?" "where is Y table written?"
#
# 7. TRANSFORMATION LOGIC — all business rules in one place.
#    Answers: "how is deduplication done?" "what filters are applied?"
#
# 8. OPERATIONAL RUNBOOK — step-by-step for support.
#    Answers: "how do I debug this?" "what do I check first?"
#
# 9. FAILURE MODES CATALOGUE — every known failure with fix.
#    Answers: "why is the raw table empty?" "what causes this error?"
#
# 10. Q&A SECTION — explicit question-answer pairs.
#     Answers: direct chatbot hits with zero retrieval ambiguity.
#
# 11. GLOSSARY — all acronyms and domain terms defined.
#     Answers: "what is EUH?" "what does deduplication mean here?"
#
# Every section repeats the source name and key entity names
# so any chunk retrieved is self-contained and interpretable.
# ─────────────────────────────────────────────────────────────────────────────

UNIFIED_DOC_PROMPT = """\
You are a senior data engineering documentation specialist building a knowledge base
that will power an AI chatbot (RAG system) for an EV data platform team.

Source System  : {source}
Generated At   : {timestamp}
Notebooks      : {notebook_list}

Your job is to synthesise the notebook summaries below into ONE comprehensive,
structured markdown document.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG WRITING RULES — FOLLOW STRICTLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER use pronouns like "it", "this", "they" to refer to tables, pipelines,
   or notebooks. Always use the full name.
   ✗ "It reads from the table and filters it."
   ✓ "The {source}_sessions notebook reads from raw.{source}_sessions and filters rows where status = 'CANCELLED'."

2. REPEAT CONTEXT — start each section with a one-line context sentence that
   names the source system and what the section covers. This ensures any
   retrieved chunk is self-contained.

3. USE EXACT NAMES — every table name, column name, function name, and
   variable name must appear exactly as in the code. No paraphrasing.

4. Q&A DENSITY — include at least 20 explicit Q&A pairs in the Q&A section.
   These are the most valuable RAG signals. Write questions exactly as a
   team member would ask the chatbot.

5. FAILURE MODES ARE CRITICAL — this is the most common chatbot query type.
   Every failure mode must include: scenario name, root cause, exact error
   symptom, detection method, and step-by-step resolution.

6. NO INFORMATION LOSS — if a notebook summary mentions a table, column,
   transformation, or business rule, it MUST appear somewhere in the final doc.
   Do not summarise away specifics.

7. LAYERED DEPTH — write each major section so that:
   - The first 2-3 sentences give the business/plain-English answer
   - The following paragraphs go deep technical
   This serves both a business reader and an engineer from the same section.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENT STRUCTURE — USE EXACTLY THIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

# {source} Data Pipeline — Complete Documentation

> **Source System:** {source} | **Platform:** EV Data Platform on Databricks
> **Last Updated:** {timestamp} | **Auto-generated by:** doc_generator_v2

---

## 1. Document Metadata

| Field            | Value                        |
|------------------|------------------------------|
| Source System    | {source}                     |
| Platform         | Databricks (Azure)           |
| Data Layers      | Landing · Raw · EUH · Curated|
| Notebooks Count  | {notebook_count}             |
| Notebooks        | {notebook_list}              |
| Generated At     | {timestamp}                  |
| Generator        | doc_generator_v2             |
| Owner            | [FILL: team/owner name]      |
| Slack Channel    | [FILL: #channel-name]        |
| Oncall Runbook   | [FILL: link]                 |

---

## 2. Quick Reference Card

_This section answers the most common quick-lookup questions about the {source} pipeline._

**What does this pipeline do (one line)?**
[one-sentence plain English answer]

**What business data does it power?**
[bullet list of reports/dashboards/decisions enabled]

**What is the expected SLA / run time?**
[FILL: expected duration and schedule]

**Who owns this pipeline?**
[FILL: team and individual owner]

**What tables does this pipeline write to?**
[list all output tables with full qualified names]

**What tables does this pipeline read from?**
[list all input tables/sources]

**What happens if this pipeline fails?**
[business impact in plain English]

**Where do I find the notebooks?**
[list notebook paths]

---

## 3. Business Overview

_This section is written for non-technical stakeholders, product managers, and leadership.
It describes the {source} pipeline in business terms without technical jargon._

### 3.1 What Is the {source} Pipeline?
[2-3 paragraph plain English explanation. What is {source}? What data does it bring?
Why does the EV platform need this data? What business questions does it answer?]

### 3.2 Business Value
[What decisions, reports, or dashboards depend on this pipeline?
What would break for the business if this stopped working?]

### 3.3 Data Freshness & SLA
[How often does this run? How fresh is the data? What is the business expectation?]

### 3.4 Business Impact of Failure
[In plain English: if this pipeline fails, what does a business user or customer experience?]

---

## 4. Architecture & End-to-End Data Flow

_This section describes how data moves through the {source} pipeline from source system
to curated layer on the EV Databricks platform._

### 4.1 High-Level Flow

```
{source} Source System
        │
        ▼
┌─────────────────────┐
│   LANDING LAYER     │  Raw ingestion from {source} API/files
│  [notebook name]    │  Tables: [exact table names]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    RAW LAYER        │  Cleansed, deduplicated, typed
│  [notebook name]    │  Tables: [exact table names]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    EUH LAYER        │  Enriched, unified, harmonised
│  [notebook name]    │  Tables: [exact table names]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  CURATED LAYER      │  Business-ready, aggregated
│  [notebook name]    │  Tables: [exact table names]
└─────────────────────┘
          │
          ▼
   Dashboards / Reports / Downstream consumers
```

### 4.2 Layer-by-Layer Explanation

**Landing Layer — {source}**
[Explain what happens in the landing layer. What is ingested? From where?
What format? What are the exact table names written?]

**Raw Layer — {source}**
[Explain what happens in the raw layer. What cleaning? What deduplication?
What schema changes? Exact table names.]

**EUH Layer — {source}**
[Explain what happens in EUH. What enrichments? What harmonisation?
What business logic is applied? Exact table names.]

**Curated Layer — {source}** _(if applicable)_
[Explain aggregations, final business-ready transformations. Exact table names.]

---

## 5. Notebook-by-Notebook Technical Reference

_This section provides a deep technical breakdown of each notebook in the {source} pipeline.
Engineers debugging or extending this pipeline should start here._

[For EACH notebook, generate a subsection like this:]

### 5.X [Notebook Name] · [Layer]

**Full Path:** `[exact workspace path]`
**Layer:** [Landing / Raw / EUH / Curated]
**Purpose:** [One precise sentence. Name the notebook, name what it does.]

#### Inputs
| Source | Type | Key Columns | Notes |
|--------|------|-------------|-------|
| [exact name] | [Delta/API/etc] | [col1, col2] | [any notes] |

#### Outputs
| Table/File | Write Mode | Key Columns | Notes |
|------------|-----------|-------------|-------|
| [exact name] | [overwrite/append/merge] | [col1, col2] | |

#### Transformation Steps (in execution order)
1. **[Step Name]** — [What it does. Name every column and table involved. State the business reason.]
2. **[Step Name]** — [...]
[continue for all steps]

#### Business Rules & Logic
- **[Rule Name]:** [Exact rule. Include values, thresholds, column names.]
- **[Rule Name]:** [...]

#### Error Handling in this Notebook
- [How nulls are handled, with column names]
- [How duplicates are handled, with dedup keys]
- [Any explicit guards or assertions]

#### Dependencies
- [Other notebooks called, with exact paths]
- [Utilities or shared libraries used]

#### Known Issues / Gotchas
- [Any quirks, edge cases, or tech debt worth noting]

---

## 6. Complete Table Catalogue

_This section lists every table read or written by the {source} pipeline.
Use this section to answer questions like "what columns does table X have?"
or "which notebook writes to table Y?"_

[For each table:]

### 6.X `[schema].[table_name]`

| Property       | Value                          |
|----------------|--------------------------------|
| Full Name      | `[schema].[table_name]`        |
| Layer          | [Landing / Raw / EUH / Curated]|
| Written By     | [notebook name]                |
| Read By        | [notebook names]               |
| Write Mode     | [overwrite / append / merge]   |
| Format         | [Delta / Parquet]              |
| Location       | [Unity Catalog Volume or path] |
| Refresh Freq   | [FILL]                         |
| Approx. Rows   | [if known]                     |

**Schema (Key Columns):**
| Column Name | Data Type | Nullable | Description | Business Meaning |
|-------------|-----------|----------|-------------|-----------------|
| [col_name]  | [type]    | [Y/N]    | [technical] | [business term] |

---

## 7. Transformation & Business Logic Reference

_This section is a centralised reference for all transformation logic and business rules
in the {source} pipeline. Engineers and support staff should use this to answer
"how does X work?" questions without reading notebook code._

### 7.1 Deduplication Strategy
[Which notebook deduplicates? On which keys? What is the business reason?
What happens to duplicates — dropped, flagged, or archived?]

### 7.2 Filtering Rules
[Every filter applied across all notebooks. Include exact column names and values.
Example: "Sessions with status = 'CANCELLED' are excluded in raw_etl_{source} before writing to raw.{source}_sessions."]

### 7.3 Join Logic
[Every join across all notebooks. Include: left table, right table, join key(s), join type, and business reason.]

### 7.4 Enrichment Logic
[Any lookup enrichments — e.g. joining station metadata. Include source table, target table, join key.]

### 7.5 Data Type Casting & Schema Enforcement
[Any explicit type casts. Column renames. Schema alignment logic.]

### 7.6 Hardcoded Values & Constants
[Any hardcoded IDs, status values, date thresholds. What do they mean?
These are critical for debugging and change management.]

---

## 8. Operational Runbook

_This section is written for the support team and on-call engineers.
It provides step-by-step guidance for operating and troubleshooting the {source} pipeline
without needing to read the notebook code._

### 8.1 How to Trigger a Manual Run
1. Go to Databricks Workspace → Jobs → [FILL: job name]
2. Set widget `source_name` = `{source}`
3. Click "Run Now"
4. Monitor progress in the Run tab
5. Check output at: `{DOC_OUTPUT_VOLUME}/{source}.md`

### 8.2 How to Force Regenerate Documentation
1. Open the doc_generator notebook
2. Set `force_regenerate` = `true`
3. Set `source_name` = `{source}`
4. Run the notebook

### 8.3 First Steps When Something Goes Wrong
1. Check the Databricks Job Run log for the {source} pipeline
2. Identify which notebook failed (look for the notebook name in the error)
3. Check the state table: `SELECT * FROM {STATE_TABLE} WHERE source = '{source}'`
4. Check the output tables listed in Section 6 for row counts and freshness
5. Refer to Section 9 (Failure Modes) for known issues and fixes

### 8.4 Key Tables to Check During Incidents
[List the most important tables to query during an incident, with suggested SQL checks.
Example: "Check row count in raw.{source}_sessions: SELECT COUNT(*) FROM raw.{source}_sessions WHERE DATE(ingested_at) = CURRENT_DATE()"]

### 8.5 Escalation Path
| Severity | When to Escalate | Who to Contact |
|----------|-----------------|----------------|
| P1 — Data missing in production | Immediately | [FILL] |
| P2 — Pipeline failing repeatedly | After 2 failed retries | [FILL] |
| P3 — Documentation stale | Next business day | [FILL] |

---

## 9. Failure Modes & Debugging Guide

_This is the most important section for support and on-call engineers.
Every known failure mode for the {source} pipeline is documented here with
root cause, exact error symptoms, and step-by-step resolution._

[For each failure mode:]

### 9.X [Failure Mode Name]

| Property     | Details |
|--------------|---------|
| Affects      | [Which notebook / layer] |
| Frequency    | [Common / Occasional / Rare] |
| Severity     | [P1 / P2 / P3] |

**Root Cause:**
[Explain why this happens. Be specific about the code path, table, or condition that causes it.]

**Symptom / Error Message:**
```
[Exact error message or observable symptom]
```

**How to Detect:**
[What to look for in logs, table counts, or job status.]

**Step-by-Step Resolution:**
1. [First thing to check/do]
2. [Second step]
3. [...]

**Prevention:**
[How to prevent this in future, if applicable.]

---

## 10. Frequently Asked Questions (Q&A)

_This section contains explicit questions and answers about the {source} pipeline.
It is designed to directly power the RAG chatbot — each Q&A is a high-confidence retrieval target._

[Generate at least 20 Q&A pairs covering all 4 audiences.
Write questions EXACTLY as a team member would ask the chatbot.
Do not use generic questions — be specific to {source} and the actual pipeline logic.]

**Technical Engineer Questions:**

**Q: What does the {source} landing notebook do?**
A: [Full specific answer naming the notebook, inputs, outputs, and key logic.]

**Q: Where is deduplication implemented in the {source} pipeline?**
A: [Exact notebook name, function name, dedup key columns.]

**Q: What columns are written to the {source} raw table?**
A: [Complete column list with types.]

**Q: How does the {source} EUH notebook enrich the data?**
A: [Specific joins, lookups, enrichments with table and column names.]

**Q: What happens if the {source} API returns an empty response?**
A: [Error handling logic for this case.]

**Q: How do I add a new field from the {source} source system?**
A: [Step-by-step guide referencing actual notebook paths.]

**Q: What is the deduplication key for {source} sessions?**
A: [Exact column names used as dedup key.]

**Support Team Questions:**

**Q: The {source} raw table is empty today. What should I check?**
A: [Step-by-step debugging starting from the landing notebook.]

**Q: How do I check if the {source} pipeline ran successfully today?**
A: [SQL query or UI step to verify.]

**Q: What is the impact if the {source} pipeline is down for 24 hours?**
A: [Business impact statement.]

**Q: How do I manually trigger the {source} pipeline?**
A: [Exact steps referencing the job name and widget values.]

**Q: Where are the {source} pipeline logs?**
A: [Exact location in Databricks UI.]

**Business / Leadership Questions:**

**Q: What is the {source} pipeline and why do we have it?**
A: [Plain English explanation, no jargon.]

**Q: Which dashboards depend on the {source} pipeline?**
A: [List of downstream consumers.]

**Q: How fresh is the {source} data in our platform?**
A: [Data freshness / SLA in business terms.]

**Q: What is the business risk if the {source} pipeline is down?**
A: [Business impact without technical language.]

**Newcomer Questions:**

**Q: I just joined the team. What is the {source} pipeline?**
A: [Friendly onboarding explanation with context.]

**Q: What does "EUH layer" mean for the {source} pipeline?**
A: [Definition of EUH + how it applies to {source} specifically.]

**Q: Where do I start if I want to understand the {source} codebase?**
A: [Recommended reading order: which notebook to start with, what to read next.]

**Q: Who should I ask if I have questions about the {source} pipeline?**
A: [FILL: team/person + Slack channel]

---

## 11. Glossary

_This section defines all acronyms, technical terms, and domain-specific vocabulary
used in the {source} pipeline documentation. The RAG chatbot uses this section
to answer "what does X mean?" questions._

| Term | Full Form | Definition in Context of {source} Pipeline |
|------|-----------|---------------------------------------------|
| EV   | Electric Vehicle | [definition] |
| EUH  | Enterprise Unified Hub | The harmonised data layer in the EV platform where data from multiple sources (including {source}) is standardised into a common schema. |
| ETL  | Extract, Transform, Load | The process used to move data from {source} through Landing → Raw → EUH layers. |
| Landing | — | The first data layer in the EV platform. Raw data from {source} is written here with minimal transformation. |
| Raw  | — | The second layer. {source} data is cleansed, deduplicated, and typed here. |
| EUH  | — | The third layer. {source} data is enriched and harmonised with other source systems here. |
| Curated | — | The final layer. Business-ready, aggregated data for dashboards and reports. |
| Unity Catalog | — | Databricks' data governance and metadata management system used to store {source} pipeline tables and documentation. |
| Delta | Delta Lake | The open-source storage format used for all {source} pipeline tables. Supports ACID transactions and time travel. |
| RAG  | Retrieval-Augmented Generation | The AI technique powering this chatbot — it retrieves relevant documentation chunks and generates answers. |
| PAT  | Personal Access Token | Authentication token used by the doc_generator to call Databricks Workspace APIs. |
| SLA  | Service Level Agreement | The expected uptime and freshness commitment for the {source} pipeline. |
| Deduplication | — | The process of removing duplicate records from {source} data, typically using [FILL: dedup key columns]. |
| {source} | — | [FILL: What is this source system? What does it do in the EV business context?] |
[Add more terms as found in the notebook summaries]

---

## 12. Change Log

_This section tracks documentation regeneration events for the {source} pipeline._

| Version | Date | Trigger | Notebooks Changed | Notes |
|---------|------|---------|------------------|-------|
| [auto-filled by generator] | {timestamp} | [hash_change / force / scheduled] | [auto-filled] | Initial generation |

---
_This document was auto-generated by `doc_generator_v2` on the EV Databricks platform.
For questions about this documentation system, contact the platform engineering team._

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Now synthesise ALL notebook summaries below into this structure.
Replace every [placeholder] with real, specific content from the summaries.
Do not leave any placeholder empty if the summaries contain the information.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOTEBOOK SUMMARIES:
{summaries}
"""


def generate_unified_doc(source: str, summaries: list[dict], timestamp: str) -> str:
    """
    Stage 2: Generate a single unified RAG-optimised document.

    WHY TWO CALLS:
    Llama 3.3 70B handles ~5000 output tokens reliably before quality degrades.
    The full unified doc is 8000-10000 tokens. Splitting into two sequential
    calls — each generating half the document — produces consistently complete,
    high-quality output without the model trailing off mid-section.

    Call A: Sections 1-7  (metadata, overview, architecture, notebook breakdown, tables, transformations)
    Call B: Sections 8-12 (runbook, failure modes, Q&A, glossary, changelog)

    The two halves are concatenated into one final document.
    """
    notebook_list  = ", ".join(r["name"] for r in summaries)
    notebook_count = len(summaries)

    # Build summaries block — preserve all detail from Stage 1
    summaries_text = "\n\n".join(
        f"{'='*60}\n"
        f"NOTEBOOK: {r['name']}\n"
        f"PATH    : {r['path']}\n"
        f"{'='*60}\n"
        f"{r['summary']}"
        for r in summaries
    )

    shared_context = f"""Source System  : {source.title()}
Generated At   : {timestamp}
Notebooks      : {notebook_list}

NOTEBOOK SUMMARIES:
{summaries_text}"""

    rag_rules = """RAG WRITING RULES (follow strictly):
- Never use pronouns like "it" or "this" to refer to tables or notebooks — always use the full name.
- Repeat the source system name in every section header.
- Use exact names from the summaries (table names, column names, function names).
- No information loss — every table, column, and transformation from the summaries must appear somewhere.
- Write each section so the first 2 sentences give the plain-English answer, followed by technical depth."""

    # ── Call A: Technical half (Sections 1–7) ────────────────────────────────
    prompt_a = f"""You are a senior data engineering documentation specialist.
Generate the FIRST HALF of a unified RAG documentation page for the {source.title()} pipeline.

{rag_rules}

{shared_context}

Generate ONLY these sections (Sections 1 through 7). Stop after Section 7.
Use this exact structure:

---

# {source.title()} Data Pipeline — Complete Documentation

> **Source System:** {source.title()} | **Platform:** EV Data Platform on Databricks
> **Last Updated:** {timestamp} | **Auto-generated by:** doc_generator_v2

---

## 1. Document Metadata
[table: source, platform, layers, notebook count, notebooks, generated at, owner placeholder, slack placeholder]

## 2. Quick Reference Card
[what it does, tables read, tables written, business impact if down, notebook paths]

## 3. Business Overview
### 3.1 What Is the {source.title()} Pipeline?
### 3.2 Business Value
### 3.3 Data Freshness & SLA
### 3.4 Business Impact of Failure

## 4. Architecture & End-to-End Data Flow
[ASCII flow diagram: {source} Source → Landing → Raw → EUH → Curated, with exact table names and notebook names at each layer]

## 5. Notebook-by-Notebook Technical Reference
[For EACH notebook: full path, layer, purpose, inputs table, outputs table, transformation steps, business rules, error handling, dependencies, known issues]

## 6. Complete Table Catalogue
[For EACH table: full name, layer, written by, read by, write mode, format, schema with all columns, data types, business meaning]

## 7. Transformation & Business Logic Reference
### 7.1 Deduplication Strategy
### 7.2 Filtering Rules
### 7.3 Join Logic
### 7.4 Enrichment Logic
### 7.5 Data Type Casting & Schema Enforcement
### 7.6 Hardcoded Values & Constants
"""

    log.info("Stage 2 Call A — generating technical sections (1-7) for '%s'...", source)
    part_a = call_llm(prompt_a, context=f"unified_doc_a_{source}", max_tokens=LLM_STAGE2_MAX_TOKENS, timeout=LLM_TIMEOUT_STAGE2)

    # ── Call B: Operational half (Sections 8–12) ──────────────────────────────
    prompt_b = f"""You are a senior data engineering documentation specialist.
Generate the SECOND HALF of a unified RAG documentation page for the {source.title()} pipeline.

{rag_rules}

{shared_context}

Generate ONLY these sections (Sections 8 through 12). Do not repeat earlier sections.
Use this exact structure:

## 8. Operational Runbook
### 8.1 How to Trigger a Manual Run
### 8.2 How to Force Regenerate Documentation
### 8.3 First Steps When Something Goes Wrong
[step-by-step: check job log → identify failed notebook → check state table → check output tables → refer to Section 9]
### 8.4 Key Tables to Check During Incidents
[list each table with a suggested COUNT/freshness SQL query]
### 8.5 Escalation Path
[P1/P2/P3 table]

## 9. Failure Modes & Debugging Guide
[For EACH failure mode: name, affected notebook/layer, frequency, severity, root cause, exact error symptom, how to detect, step-by-step resolution, prevention]
Minimum 5 failure modes. Include at least one per pipeline layer.

## 10. Frequently Asked Questions (Q&A)
[Minimum 20 Q&A pairs. Write questions exactly as a team member would ask the chatbot.
Cover all 4 audiences: engineers, support, business, newcomers.
Reference exact table names, column names, and notebook names in answers.]

## 11. Glossary
[Table: Term | Full Form | Definition in context of {source.title()} pipeline.
Include: EV, EUH, ETL, Landing, Raw, Curated, Unity Catalog, Delta, RAG, PAT, SLA, Deduplication, {source.title()}, and every domain term from the summaries.]

## 12. Change Log
| Version | Date | Trigger | Notebooks Changed | Notes |
|---------|------|---------|------------------|-------|
| 1.0 | {timestamp} | initial_generation | {notebook_list} | Auto-generated |

---
_Auto-generated by doc_generator_v2. For questions contact the platform engineering team._
"""

    log.info("Stage 2 Call B — generating operational sections (8-12) for '%s'...", source)
    part_b = call_llm(prompt_b, context=f"unified_doc_b_{source}", max_tokens=LLM_STAGE2_MAX_TOKENS, timeout=LLM_TIMEOUT_STAGE2)

    # ── Concatenate both halves ────────────────────────────────────────────────
    log.info("Stage 2 complete — concatenating halves for '%s'", source)
    return part_a.rstrip() + "\n\n" + part_b.lstrip()


# COMMAND ----------
# MAGIC %md ## Cell 9 — Hashing & State Management

# COMMAND ----------

from pyspark.sql.types import StringType, StructField, StructType, TimestampType

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
        source         STRING    NOT NULL,
        hash           STRING    NOT NULL,
        notebook_count STRING,
        updated_at     TIMESTAMP NOT NULL,
        generated_by   STRING
    )
    USING DELTA
    COMMENT 'Tracks doc generation state per source. Used for hash-based change detection.'
""")


def compute_hash(texts: list[str]) -> str:
    """SHA-256 over all notebook contents. Separator prevents boundary collisions."""
    combined = "||".join(texts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def get_stored_hash(source: str) -> Optional[str]:
    rows = (
        spark.table(STATE_TABLE)
        .filter(F.col("source") == source)
        .select("hash")
        .collect()
    )
    return rows[0]["hash"] if rows else None


def save_state(source: str, new_hash: str, notebook_count: int) -> None:
    """MERGE so other sources' state is never overwritten."""
    STATE_SCHEMA = StructType([
        StructField("source",         StringType(),    False),
        StructField("hash",           StringType(),    False),
        StructField("notebook_count", StringType(),    True),
        StructField("updated_at",     TimestampType(), False),
        StructField("generated_by",   StringType(),    True),
    ])

    new_row = spark.createDataFrame(
        [(source, new_hash, str(notebook_count), datetime.now(timezone.utc), "doc_generator_v2")],
        schema=STATE_SCHEMA,
    )
    new_row.createOrReplaceTempView("_doc_state_update")

    spark.sql(f"""
        MERGE INTO {STATE_TABLE} AS target
        USING _doc_state_update AS source
        ON target.source = source.source
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    log.info("State saved for '%s' (hash: %s…)", source, new_hash[:12])

# COMMAND ----------
# MAGIC %md ## Cell 10 — Output Writer

# COMMAND ----------

def write_doc(source: str, content: str) -> str:
    """
    Write the unified documentation to Unity Catalog Volume.
    Single file: {source}.md

    Returns the output path.
    """
    dbutils.fs.mkdirs(DOC_OUTPUT_VOLUME)
    output_path = f"{DOC_OUTPUT_VOLUME}/{source}.md"
    dbutils.fs.put(output_path, content, overwrite=True)
    log.info("Documentation written: %s", output_path)
    return output_path

# COMMAND ----------
# MAGIC %md ## Cell 11 — Main Execution

# COMMAND ----------

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

log.info("=" * 60)
log.info("DOC GENERATOR v2.0 — START  source=%s  force=%s", source, force_regenerate)
log.info("=" * 60)

# ── Step 1: Export notebooks ──────────────────────────────────────────────────
log.info("Step 1/5 — Exporting %d notebooks…", len(workflow_paths))
contents = [export_notebook(p) for p in workflow_paths]

valid_paths    = [p for p, c in zip(workflow_paths, contents) if c is not None]
valid_contents = [c for c in contents if c is not None]

failed_exports = len(workflow_paths) - len(valid_paths)
if failed_exports:
    log.warning("%d notebook(s) failed to export and will be skipped.", failed_exports)

if not valid_contents:
    raise RuntimeError("❌ All notebook exports failed. Check PAT token permissions.")

# ── Step 2: Hash check ────────────────────────────────────────────────────────
log.info("Step 2/5 — Checking content hash…")
current_hash = compute_hash(valid_contents)
stored_hash  = get_stored_hash(source)

log.info("  Current : %s…", current_hash[:16])
log.info("  Stored  : %s…", (stored_hash or "none")[:16])

if stored_hash == current_hash and not force_regenerate:
    log.info("✅ No changes detected for '%s'. Skipping.", source)
    dbutils.notebook.exit(json.dumps({
        "status": "skipped",
        "reason": "no_changes",
        "source": source,
        "hash": current_hash,
    }))

# ── Step 3: Stage 1 — Per-notebook summarisation ──────────────────────────────
log.info("Step 3/5 — Stage 1: Summarising %d notebooks…", len(valid_paths))
notebook_results = summarize_all_notebooks(valid_paths, valid_contents)

failed_summaries = [r for r in notebook_results if r["error"]]
if failed_summaries:
    log.warning(
        "%d/%d summaries had errors (included with error markers).",
        len(failed_summaries), len(notebook_results)
    )

# ── Step 4: Stage 2 — Unified RAG document ───────────────────────────────────
log.info("Step 4/5 — Stage 2: Generating unified RAG document…")
unified_doc = generate_unified_doc(source, notebook_results, timestamp)

# ── Step 5: Write & save state ────────────────────────────────────────────────
log.info("Step 5/5 — Writing output and saving state…")
output_path = write_doc(source, unified_doc)
save_state(source, current_hash, len(valid_paths))

# ── Final summary ─────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("✅ COMPLETE  source='%s'  notebooks=%d  output=%s",
         source, len(valid_paths), output_path)
log.info("=" * 60)

result = {
    "status":               "success",
    "source":               source,
    "notebooks_processed":  len(valid_paths),
    "notebooks_skipped":    failed_exports,
    "output_path":          output_path,
    "hash":                 current_hash,
    "generated_at":         datetime.now(timezone.utc).isoformat(),
}

print(json.dumps(result, indent=2))
dbutils.notebook.exit(json.dumps(result))
