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
# Split model strategy:
#   Stage 1 — per-notebook summarisation. Runs N times (once per notebook).
#             Cost matters. Llama 3.3 70B is fast, cheap, and good enough
#             for structured extraction tasks.
#
#   Stage 2 — unified RAG doc generation. Runs once per source.
#             Quality is everything here — this doc powers the chatbot.
#             Llama 3.1 405B has significantly stronger instruction following
#             for long, complex, multi-rule prompts.
#             Token limit raised to 8000 — the unified doc needs room.

# Stage 1 — per-notebook summary (cost-optimised)
LLM_STAGE1_ENDPOINT   = "databricks-meta-llama-3-3-70b-instruct"
LLM_STAGE1_MAX_TOKENS = 2000

# Stage 2 — unified RAG document (quality-optimised)
LLM_STAGE2_ENDPOINT   = "databricks-meta-llama-3-1-405b-instruct"
LLM_STAGE2_MAX_TOKENS = 8000

# Shared
LLM_TEMPERATURE   = 0.1
LLM_RETRY_LIMIT   = 3
LLM_RETRY_BACKOFF = 2.0

# ── Processing Config ─────────────────────────────────────────────────────────
MAX_CODE_CHARS     = 20_000
MAX_PARALLEL_CALLS = 4

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
    endpoint: str = LLM_STAGE1_ENDPOINT,
    max_tokens: int = LLM_STAGE1_MAX_TOKENS,
) -> str:
    """
    Call Databricks model serving with exponential backoff retry.
    Distinguishes retryable (5xx, 429) from non-retryable (4xx) errors.

    Args:
        prompt:     The user prompt.
        context:    Label for logging (e.g. notebook name).
        endpoint:   Model serving endpoint name. Defaults to Stage 1 (70B).
        max_tokens: Max tokens for this call. Stage 1=2000, Stage 2=8000.
    """
    url     = f"{WORKSPACE_URL}/serving-endpoints/{endpoint}/invocations"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    log.debug("LLM → endpoint=%s  max_tokens=%d  context=%s", endpoint, max_tokens, context)

    last_error = None
    for attempt in range(1, LLM_RETRY_LIMIT + 1):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        except requests.HTTPError as e:
            if r.status_code < 500 and r.status_code != 429:
                raise
            last_error = e
            wait = LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning("LLM attempt %d failed: %s — retrying in %.1fs", attempt, e, wait)
            time.sleep(wait)

        except Exception as e:
            last_error = e
            wait = LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning("LLM attempt %d error: %s — retrying in %.1fs", attempt, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"LLM failed after {LLM_RETRY_LIMIT} attempts [{context}]: {last_error}")

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

Extract and document the following. Be exhaustive — do not skip anything.
Always use the exact names from the code (table names, column names, function names, variable names).

---

**1. NOTEBOOK PURPOSE**
One precise sentence describing what this notebook does and why it exists.

**2. DATA LAYER**
Which layer does this notebook belong to? (Landing / Raw / EUH / Curated)
Why does it belong there?

**3. INPUTS**
List every data source read:
- Table/file/API name (exact name as in code)
- Format (Delta, Parquet, API, etc.)
- Key columns used

**4. OUTPUTS**
List every table or file written:
- Table/file name (exact name as in code)
- Format
- Write mode (overwrite, append, merge)
- Key columns written

**5. TRANSFORMATIONS (step by step)**
List every transformation in execution order:
- Step name
- What it does
- Columns involved
- Business reason (if inferrable)

**6. BUSINESS LOGIC & RULES**
Document every business rule embedded in the code:
- Conditions, thresholds, filters with their exact values
- Deduplication keys and strategies
- Join conditions and their purpose
- Any hardcoded values (IDs, statuses, dates) and what they mean

**7. ERROR HANDLING & DATA QUALITY**
- How are nulls handled?
- How are duplicates handled?
- Are there any explicit checks, assertions, or guards?
- What happens if input data is empty or malformed?

**8. DEPENDENCIES**
- Other notebooks called (exact paths)
- Shared utilities or functions imported
- External APIs or services called
- Jobs or clusters expected to run first

**9. FAILURE MODES**
List every realistic way this notebook can fail:
- Failure scenario name
- Root cause
- Symptom (what the error looks like)
- How to detect it
- How to fix it

**10. PERFORMANCE NOTES**
- Any caching, broadcasting, or repartitioning
- Potential bottlenecks
- Approximate data volume handled (if inferrable)

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


def summarize_notebook(path: str, raw_content: str) -> dict:
    name = path.split("/")[-1]
    code = clean_code(raw_content)[:MAX_CODE_CHARS]

    if len(clean_code(raw_content)) > MAX_CODE_CHARS:
        log.warning("Notebook '%s' truncated to %d chars", name, MAX_CODE_CHARS)

    prompt = NOTEBOOK_SUMMARY_PROMPT.format(
        source=source.title(),
        name=name,
        path=path,
        code=code,
    )

    try:
        # Stage 1 — use 70B (fast, cost-optimised)
        summary = call_llm(
            prompt,
            context=name,
            endpoint=LLM_STAGE1_ENDPOINT,
            max_tokens=LLM_STAGE1_MAX_TOKENS,
        )
        log.info("✓ Summarised: %s", name)
        return {"path": path, "name": name, "summary": summary, "error": None}
    except Exception as e:
        log.error("✗ Failed: %s — %s", name, e)
        return {
            "path": path, "name": name,
            "summary": f"_Summary unavailable: {e}_", "error": str(e),
        }


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
    Stage 2: Generate a single unified RAG-optimised document from all notebook summaries.
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

    prompt = UNIFIED_DOC_PROMPT.format(
        source=source.title(),
        timestamp=timestamp,
        notebook_list=notebook_list,
        notebook_count=notebook_count,
        DOC_OUTPUT_VOLUME=DOC_OUTPUT_VOLUME,
        STATE_TABLE=STATE_TABLE,
        summaries=summaries_text,
    )

    log.info("Generating unified RAG document for source '%s'…", source)
    # Stage 2 — use 405B (quality-optimised, runs only once per source)
    return call_llm(
        prompt,
        context=f"unified_doc_{source}",
        endpoint=LLM_STAGE2_ENDPOINT,
        max_tokens=LLM_STAGE2_MAX_TOKENS,
    )

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
