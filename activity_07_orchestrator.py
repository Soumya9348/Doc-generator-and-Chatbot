# Databricks notebook source
# MAGIC %md
# MAGIC # 🧭 Activity 7: Copilot Orchestrator (Knowledge Agent + Genie)
# MAGIC
# MAGIC **Single notebook** containing the full orchestrator:
# MAGIC 1. Genie reporting views setup
# MAGIC 2. Intent classifier
# MAGIC 3. Knowledge Agent (structured-first + vector fallback)
# MAGIC 4. Genie Space integration
# MAGIC 5. Orchestrator (routes to correct agent)
# MAGIC 6. Conversation logging
# MAGIC
# MAGIC ```
# MAGIC User Query → Intent Classifier → ┬→ Knowledge Agent  → Answer
# MAGIC                                   ├→ Genie Space       → Answer
# MAGIC                                   └→ Both (HYBRID)     → Merged Answer
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration

# COMMAND ----------

import json
import re
import os
import uuid
import time
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import mlflow.deployments
from databricks.vector_search.client import VectorSearchClient

CONFIG = {
    # LLM
    "llm_endpoint": "databricks-claude-sonnet-4-6",

    # Embedding
    "embedding_endpoint": "databricks-gte-large-en",

    # Knowledge base
    "knowledge_table": "emobility-uc-dev.sandbox-emobility.copilot_knowledge_chunks",
    "conversations_table": "emobility-uc-dev.sandbox-emobility.copilot_conversations",

    # Vector Search (pre-prod)
    "vs_endpoint": "copilot-vs-endpoint",
    "vs_index": "emobility-uc-dev.sandbox-emobility.copilot_knowledge_index",

    # Genie Space — UPDATE with your actual Space ID
    "genie_space_id": "PUT_YOUR_GENIE_SPACE_ID_HERE",

    # Retrieval settings
    "structured_min_chunks": 1,
    "vector_top_k": 5,
    "max_chunks_for_synthesis": 3,

    # Sources
    "known_sources": ["driivz", "enovos", "spirii", "uberall"],
}

deploy_client = mlflow.deployments.get_deploy_client("databricks")
vsc = VectorSearchClient(disable_notice=True)

print("✅ Configuration loaded")
print(f"   LLM:       {CONFIG['llm_endpoint']}")
print(f"   Embedding: {CONFIG['embedding_endpoint']}")
print(f"   VS Index:  {CONFIG['vs_index']}")
print(f"   Genie:     {CONFIG['genie_space_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part A: Genie Reporting Views
# MAGIC
# MAGIC Creates views with business-friendly column descriptions so Genie can answer data questions.
# MAGIC
# MAGIC **📌 Adjust column names if they don't match your EUH tables.**

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View 1: Charging Locations
# MAGIC CREATE OR REPLACE VIEW `emobility-uc-dev`.`sandbox-emobility`.v_genie_charger_locations
# MAGIC (
# MAGIC   source            COMMENT 'Source system (driivz, spirii, uberall, enovos)',
# MAGIC   source_location_id COMMENT 'Unique ID of the charging location in the source system',
# MAGIC   location_name     COMMENT 'Human-readable name of the charging station/location',
# MAGIC   operator          COMMENT 'Operator/company managing this charging location',
# MAGIC   owning_company    COMMENT 'Company that owns this charging location',
# MAGIC   country_code      COMMENT 'ISO country code (e.g., DE, NL, DK)',
# MAGIC   city              COMMENT 'City where the location is situated',
# MAGIC   address           COMMENT 'Street address of the charging location',
# MAGIC   postal_code       COMMENT 'Postal/ZIP code',
# MAGIC   latitude          COMMENT 'GPS latitude',
# MAGIC   longitude         COMMENT 'GPS longitude',
# MAGIC   status            COMMENT 'Current status (active, inactive)',
# MAGIC   location_type     COMMENT 'Type of location (e.g., Shell Recharge)',
# MAGIC   created           COMMENT 'When this record was first created'
# MAGIC )
# MAGIC AS
# MAGIC SELECT source, source_location_id, name AS location_name, operator, owning_company,
# MAGIC        country_code, city, address, postal_code, latitude, longitude,
# MAGIC        status, location_type, created
# MAGIC FROM `emobility-uc-dev`.`euh`.charger_location;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View 2: EVSE / Charging Points
# MAGIC CREATE OR REPLACE VIEW `emobility-uc-dev`.`sandbox-emobility`.v_genie_charger_evse
# MAGIC (
# MAGIC   source            COMMENT 'Source system (driivz, spirii, uberall, enovos)',
# MAGIC   location_id       COMMENT 'Internal location ID this EVSE belongs to',
# MAGIC   source_location_id COMMENT 'Location ID in the source system',
# MAGIC   source_evse_id    COMMENT 'Unique EVSE ID in the source system',
# MAGIC   chargepoint_id    COMMENT 'Charge point identifier',
# MAGIC   latitude          COMMENT 'GPS latitude of the EVSE',
# MAGIC   longitude         COMMENT 'GPS longitude of the EVSE',
# MAGIC   created           COMMENT 'When this EVSE record was created',
# MAGIC   modified          COMMENT 'When this EVSE was last modified'
# MAGIC )
# MAGIC AS
# MAGIC SELECT source, location_id, source_location_id, source_evse_id,
# MAGIC        chargepoint_id, latitude, longitude, created, modified
# MAGIC FROM `emobility-uc-dev`.`euh`.charger_evse;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View 3: Connectors
# MAGIC CREATE OR REPLACE VIEW `emobility-uc-dev`.`sandbox-emobility`.v_genie_charger_connectors
# MAGIC (
# MAGIC   source              COMMENT 'Source system (driivz, spirii, uberall, enovos)',
# MAGIC   source_connector_id COMMENT 'Unique connector ID in the source system',
# MAGIC   source_evse_id      COMMENT 'EVSE ID this connector belongs to',
# MAGIC   connector_type      COMMENT 'Type of connector (e.g., Type 2, CCS, CHAdeMO)',
# MAGIC   power_type          COMMENT 'Power type: AC or DC',
# MAGIC   power_kw            COMMENT 'Maximum power output in kilowatts (kW)',
# MAGIC   phase               COMMENT 'Number of electrical phases (1 or 3)'
# MAGIC )
# MAGIC AS
# MAGIC SELECT source, source_connector_id, source_evse_id,
# MAGIC        connector_type, power_type, power_kw, phase
# MAGIC FROM `emobility-uc-dev`.`euh`.charger_connector;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View 4: Charging Sessions (UNCOMMENT if charger_session table exists)
# MAGIC
# MAGIC -- CREATE OR REPLACE VIEW `emobility-uc-dev`.`sandbox-emobility`.v_genie_charger_sessions
# MAGIC -- (
# MAGIC --   source                    COMMENT 'Source system (driivz)',
# MAGIC --   source_session_id         COMMENT 'Unique session ID in the source system',
# MAGIC --   location_id               COMMENT 'Location where the session occurred',
# MAGIC --   evse_id                   COMMENT 'EVSE used for this session',
# MAGIC --   connector_id              COMMENT 'Connector used for this session',
# MAGIC --   session_start             COMMENT 'When the charging session started',
# MAGIC --   session_end               COMMENT 'When the charging session ended',
# MAGIC --   session_duration_seconds  COMMENT 'Total session duration in seconds (includes idle time)',
# MAGIC --   charging_duration_seconds COMMENT 'Actual charging duration in seconds (energy flowing)',
# MAGIC --   energy_kwh               COMMENT 'Total energy delivered in kWh',
# MAGIC --   status                    COMMENT 'Session status (completed, failed, etc.)'
# MAGIC -- )
# MAGIC -- AS
# MAGIC -- SELECT source, source_session_id, location_id, evse_id, connector_id,
# MAGIC --        session_start, session_end, session_duration_seconds,
# MAGIC --        charging_duration_seconds, energy_kwh, status
# MAGIC -- FROM `emobility-uc-dev`.`euh`.charger_session;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify views
# MAGIC SHOW VIEWS IN `emobility-uc-dev`.`sandbox-emobility` LIKE 'v_genie_*';

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part B: LLM Helper + Prompts

# COMMAND ----------

def call_llm(system_prompt: str, user_message: str, max_tokens: int = 500, temperature: float = 0) -> str:
    """Call Claude endpoint and return response text."""
    response = deploy_client.predict(
        endpoint=CONFIG["llm_endpoint"],
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    )
    return response["choices"][0]["message"]["content"]

def parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    json_str = raw.strip()
    if json_str.startswith("```"):
        json_str = re.sub(r'^```(?:json)?\n?', '', json_str)
        json_str = re.sub(r'\n?```$', '', json_str)
    return json.loads(json_str)


print("✅ LLM helper defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part C: Intent Classifier

# COMMAND ----------

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for the eMobility DataPlatform Copilot.

Classify the user's query into ONE of these categories:

KNOWLEDGE_LOOKUP — Questions about documentation, architecture, how pipelines work, transformation logic, business rules, column definitions, join logic, error handling, source system descriptions.
Examples: "How does Driivz EUH transform sessions?", "What are the business rules for Spirii?", "What deduplication logic is used?"

STRUCTURED_QUERY — Questions that need data/numbers from the database. Metrics, counts, totals, averages, trends, comparisons, "how many", "show me", "list all".
Examples: "How many active stations are there?", "What was the total revenue last month?", "Show sessions by country"

HYBRID — Questions that need BOTH data AND explanation. Typically "why" questions or questions that need context alongside numbers.
Examples: "Why did session count drop last week?", "What's the trend and how does the pipeline handle it?"

Return ONLY valid JSON:
{"intent": "KNOWLEDGE_LOOKUP|STRUCTURED_QUERY|HYBRID", "confidence": 0.XX, "reasoning": "brief reason"}
"""


def classify_intent(user_query: str) -> dict:
    """Classify: KNOWLEDGE_LOOKUP, STRUCTURED_QUERY, or HYBRID."""
    try:
        raw = call_llm(INTENT_CLASSIFIER_PROMPT, user_query, max_tokens=100, temperature=0)
        parsed = parse_llm_json(raw)
        return {
            "intent": parsed.get("intent", "KNOWLEDGE_LOOKUP"),
            "confidence": parsed.get("confidence", 0.5),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        print(f"   ⚠️ Intent classification error: {e}")
        return {"intent": "KNOWLEDGE_LOOKUP", "confidence": 0.5, "reasoning": "fallback"}


print("✅ Intent Classifier defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part D: Knowledge Agent (Structured-First + Vector Fallback)

# COMMAND ----------

QUERY_UNDERSTANDING_PROMPT = """You are a query parser for the eMobility DataPlatform.

Given a user query, extract structured metadata for deterministic document retrieval.

Known source systems: driivz, enovos, spirii, uberall
Known data layers: landing, raw, euh
Known section types: path, notebook_purpose, data_layer, column_transformations, transformation_steps, business_rules, deduplication_logic, join_logic, error_handling, source_overview, table_details

Return ONLY valid JSON:
{"source_name": "<or null>", "data_layer": "<landing|raw|euh or null>", "section_type": "<or null>", "tables_mentioned": [], "search_terms": [], "confidence": 0.XX}
"""

RESPONSE_COMPOSER_PROMPT = """You are the Response Composer for the eMobility DataPlatform Copilot.

Take the retrieved documentation chunks and synthesize a clear, accurate answer.

Rules:
1. Answer directly — don't repeat the question
2. Use ONLY the provided chunks — never make up information
3. Cite sources: [Source: notebook_name §section_name]
4. If chunks don't fully answer, say what you know and what's missing
5. Use markdown formatting (headers, bullets, code blocks)
6. Keep concise but complete (200-400 words)
7. Include column names and logic when discussing transformations
"""

# ─── Query Understanding ───
def understand_query(user_query: str) -> dict:
    try:
        raw = call_llm(QUERY_UNDERSTANDING_PROMPT, user_query, max_tokens=200, temperature=0)
        return parse_llm_json(raw)
    except:
        return {"source_name": None, "data_layer": None, "section_type": None,
                "tables_mentioned": [], "search_terms": user_query.lower().split(), "confidence": 0.3}


# ─── Structured Retrieval ───
def structured_retrieval(metadata: dict) -> list[dict]:
    conditions = []
    if metadata.get("source_name"):
        conditions.append(f"source_name = '{metadata['source_name']}'")
    if metadata.get("data_layer"):
        conditions.append(f"data_layer = '{metadata['data_layer']}'")
    if metadata.get("section_type"):
        conditions.append(f"section_header = '{metadata['section_type']}'")
    if metadata.get("tables_mentioned"):
        tbl_conds = [f"(array_contains(tables_mentioned, '{t}') OR LOWER(content) LIKE '%{t.lower()}%')"
                     for t in metadata["tables_mentioned"]]
        conditions.append(f"({' OR '.join(tbl_conds)})")
    if metadata.get("search_terms") and not conditions:
        kw_conds = [f"LOWER(content) LIKE '%{t.lower()}%'" for t in metadata["search_terms"][:3]]
        conditions.append(f"({' OR '.join(kw_conds)})")

    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"""SELECT chunk_id, content, source_name, notebook_name, data_layer, section_header
                FROM {CONFIG['knowledge_table']} WHERE {where}
                ORDER BY chunk_index LIMIT 5"""
    try:
        rows = spark.sql(query).collect()
        return [{"chunk_id": r.chunk_id, "content": r.content, "source_name": r.source_name or "",
                 "notebook_name": r.notebook_name or "", "data_layer": r.data_layer or "",
                 "section_header": r.section_header or "", "method": "structured", "score": 1.0}
                for r in rows]
    except Exception as e:
        print(f"   ❌ Structured retrieval error: {e}")
        return []


# ─── Vector Search Fallback ───
def embed_query(query: str) -> list[float]:
    response = deploy_client.predict(endpoint=CONFIG["embedding_endpoint"], inputs={"input": [query]})
    return response["data"][0]["embedding"]


def vector_search(user_query: str, metadata: dict) -> list[dict]:
    filters = {}
    if metadata.get("source_name"):
        filters["source_name"] = metadata["source_name"]
    if metadata.get("data_layer"):
        filters["data_layer"] = metadata["data_layer"]

    try:
        index = vsc.get_index(endpoint_name=CONFIG["vs_endpoint"], index_name=CONFIG["vs_index"])
        kwargs = {"query_vector": embed_query(user_query),
                  "columns": ["chunk_id", "content", "source_name", "notebook_name", "data_layer", "section_header"],
                  "num_results": CONFIG["vector_top_k"]}
        if filters:
            kwargs["filters"] = filters
        results = index.similarity_search(**kwargs)
        chunks = []
        if results and "result" in results and results["result"]["data_array"]:
            for row in results["result"]["data_array"]:
                chunks.append({"chunk_id": row[0] or "", "content": row[1] or "",
                               "source_name": row[2] or "", "notebook_name": row[3] or "",
                               "data_layer": row[4] or "", "section_header": row[5] or "",
                               "method": "vector", "score": 0.8})
        return chunks
    except Exception as e:
        print(f"   ❌ Vector search error: {e}")
        return []


# ─── Response Composer ───
def compose_response(user_query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"answer": "I don't have documentation covering this topic. Try asking about driivz, enovos, spirii, or uberall.",
                "citations": [], "method": "none"}

    seen, unique = set(), []
    for c in chunks:
        key = c["content"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    top = unique[:CONFIG["max_chunks_for_synthesis"]]

    citations = []
    chunks_text = ""
    for i, c in enumerate(top):
        label = f"{c['notebook_name'] or c['source_name']} §{c['section_header']}"
        citations.append({"source": label, "method": c["method"], "layer": c["data_layer"]})
        chunks_text += f"\n--- Chunk {i+1} [Source: {label}] [Layer: {c['data_layer']}] ---\n{c['content']}\n"

    prompt = f"User Question: {user_query}\n\nRetrieved Documentation:\n{chunks_text}\n\nSynthesize a clear answer with [Source: ...] citations."

    try:
        answer = call_llm(RESPONSE_COMPOSER_PROMPT, prompt, max_tokens=800, temperature=0)
    except Exception as e:
        answer = f"Error: {e}\n\nRaw chunks:\n"
        for c in top:
            answer += f"\n**[{c['notebook_name']} §{c['section_header']}]**\n{c['content'][:300]}\n"

    return {"answer": answer, "citations": citations, "method": top[0]["method"] if top else "none"}


# ─── Knowledge Agent Pipeline ───
def knowledge_agent(user_query: str, verbose: bool = True) -> dict:
    """Full pipeline: understand → structured → [vector fallback] → compose."""
    metadata = understand_query(user_query)
    if verbose:
        print(f"   🏷️ Metadata: source={metadata.get('source_name')}, layer={metadata.get('data_layer')}, section={metadata.get('section_type')}")

    structured = structured_retrieval(metadata)
    if verbose:
        print(f"   🗄️ Structured: {len(structured)} chunks")

    all_chunks = structured
    method = "structured"

    if len(structured) < CONFIG["structured_min_chunks"]:
        vector = vector_search(user_query, metadata)
        seen_ids = {c["chunk_id"] for c in structured}
        for vc in vector:
            if vc["chunk_id"] not in seen_ids:
                all_chunks.append(vc)
                seen_ids.add(vc["chunk_id"])
        method = "hybrid" if structured else "vector"
        if verbose:
            print(f"   🔍 Vector fallback: {len(vector)} chunks → total: {len(all_chunks)}")

    result = compose_response(user_query, all_chunks)
    result["retrieval_method"] = method
    result["metadata"] = metadata
    return result


print("✅ Knowledge Agent defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part E: Genie Space Integration

# COMMAND ----------

def genie_query(user_query: str, verbose: bool = True) -> dict:
    """
    Route a data question to Genie Space via the Databricks SDK.
    Handles async polling for results.
    """
    space_id = CONFIG["genie_space_id"]

    if space_id == "PUT_YOUR_GENIE_SPACE_ID_HERE":
        return {
            "answer": "⚠️ **Genie Space not configured.** Update `CONFIG['genie_space_id']` with your Space ID.\n\n"
                      "Find it in: **Databricks UI → Genie → your space → URL contains the ID.**",
            "citations": [{"source": "Genie Space", "method": "genie", "layer": "reporting"}],
            "retrieval_method": "genie",
            "sql_generated": None,
        }

    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        if verbose:
            print(f"   📊 Sending to Genie Space: {space_id}")

        # Start conversation
        response = w.genie.start_conversation(space_id=space_id, content=user_query)
        conversation_id = response.conversation_id
        message_id = response.message_id

        # Poll for completion
        max_wait, waited = 120, 0
        result = None

        while waited < max_wait:
            msg = w.genie.get_message(
                space_id=space_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
            status = msg.status.value if hasattr(msg.status, 'value') else str(msg.status)

            if status in ("COMPLETED", "COMPLETED_WITH_ERROR"):
                result = msg
                break
            elif status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
                return {"answer": f"Genie query failed: {status}", "citations": [],
                        "retrieval_method": "genie", "sql_generated": None}

            if verbose and waited % 10 == 0:
                print(f"   ⏳ Genie status: {status} ({waited}s)")
            time.sleep(3)
            waited += 3

        if result is None:
            return {"answer": "Genie query timed out.", "citations": [],
                    "retrieval_method": "genie", "sql_generated": None}

        # Extract answer
        answer_text = ""
        sql_generated = None

        for attachment in (result.attachments or []):
            if hasattr(attachment, 'text') and attachment.text:
                answer_text += attachment.text.content + "\n"

            if hasattr(attachment, 'query') and attachment.query:
                sql_generated = attachment.query.query

                if hasattr(attachment.query, 'result') and attachment.query.result:
                    res = attachment.query.result
                    columns = [col.name for col in (res.columns or [])]
                    data = res.data_array or []

                    if columns and data:
                        answer_text += "\n| " + " | ".join(columns) + " |\n"
                        answer_text += "| " + " | ".join(["---"] * len(columns)) + " |\n"
                        for row in data[:20]:
                            answer_text += "| " + " | ".join(str(v or "") for v in row) + " |\n"

        return {
            "answer": answer_text.strip() or "Query completed but no text returned.",
            "citations": [{"source": "Genie Space", "method": "genie", "layer": "reporting"}],
            "retrieval_method": "genie",
            "sql_generated": sql_generated,
        }

    except ImportError:
        return {"answer": "⚠️ `databricks-sdk` not available.", "citations": [],
                "retrieval_method": "genie", "sql_generated": None}
    except Exception as e:
        return {"answer": f"Genie error: {str(e)}", "citations": [],
                "retrieval_method": "genie", "sql_generated": None}


print("✅ Genie Space integration defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part F: The Orchestrator

# COMMAND ----------

class CopilotOrchestrator:
    """
    Main orchestrator — classifies intent, routes to agents, logs conversations.
    """

    def __init__(self):
        self.conversation_id = str(uuid.uuid4())
        self.turn_number = 0
        self.history = []

    def query(self, user_query: str, user_role: str = "Engineer", verbose: bool = True) -> dict:
        """Process a user query through the full pipeline."""
        self.turn_number += 1
        start_time = time.time()

        # ─── Step 1: Intent ───
        intent_result = classify_intent(user_query)
        intent = intent_result["intent"]

        if verbose:
            print(f"\n{'='*70}")
            print(f"💬 [{self.turn_number}] \"{user_query}\"")
            print(f"   🎯 Intent: {intent} ({intent_result['confidence']:.2f}) — {intent_result['reasoning']}")

        # ─── Step 2: Route ───
        result = None

        if intent == "KNOWLEDGE_LOOKUP":
            if verbose:
                print(f"   📚 Routing → Knowledge Agent")
            result = knowledge_agent(user_query, verbose=verbose)

        elif intent == "STRUCTURED_QUERY":
            if verbose:
                print(f"   📊 Routing → Genie Space")
            result = genie_query(user_query, verbose=verbose)

        elif intent == "HYBRID":
            if verbose:
                print(f"   🔗 Routing → Knowledge Agent + Genie Space")
            k_result = knowledge_agent(user_query, verbose=verbose)
            g_result = genie_query(user_query, verbose=verbose)

            merged = ""
            if k_result.get("answer"):
                merged += "### 📚 From Documentation\n\n" + k_result["answer"] + "\n\n"
            if g_result.get("answer") and "not configured" not in g_result.get("answer", ""):
                merged += "### 📊 From Data\n\n" + g_result["answer"]

            result = {
                "answer": merged or k_result.get("answer", "No results."),
                "citations": k_result.get("citations", []) + g_result.get("citations", []),
                "retrieval_method": "hybrid",
                "sql_generated": g_result.get("sql_generated"),
            }
        else:
            result = knowledge_agent(user_query, verbose=verbose)

        latency_ms = int((time.time() - start_time) * 1000)

        # ─── Enrich ───
        result["intent"] = intent
        result["intent_confidence"] = intent_result["confidence"]
        result["latency_ms"] = latency_ms
        result["turn_number"] = self.turn_number
        result["conversation_id"] = self.conversation_id

        # ─── History ───
        self.history.append({"turn": self.turn_number, "query": user_query, "intent": intent})
        if len(self.history) > 5:
            self.history = self.history[-5:]

        # ─── Log ───
        self._log_conversation(user_query, result, user_role, latency_ms)

        # ─── Display ───
        if verbose:
            print(f"\n{'─'*70}")
            print(f"📝 Answer ({result['retrieval_method'].upper()} | ⏱️ {latency_ms}ms)")
            print(f"{'─'*70}")
            print(result["answer"][:1500])
            if len(result["answer"]) > 1500:
                print(f"\n... [{len(result['answer']) - 1500} more chars]")
            if result.get("citations"):
                print(f"\n📎 Citations:")
                for c in result["citations"]:
                    print(f"   [{c['method'].upper()}] {c['source']}")
            if result.get("sql_generated"):
                print(f"\n💾 SQL:\n{result['sql_generated']}")

        return result

    def _log_conversation(self, user_query, result, user_role, latency_ms):
        """Log turn to conversations table."""
        try:
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, ArrayType
            log = [{
                "conversation_id": self.conversation_id,
                "turn_number": self.turn_number,
                "user_query": user_query,
                "intent_classified": result.get("intent", ""),
                "agent_used": result.get("retrieval_method", ""),
                "response_text": result.get("answer", "")[:5000],
                "retrieval_method": result.get("retrieval_method", ""),
                "sources_used": [c["source"] for c in result.get("citations", [])],
                "sql_generated": result.get("sql_generated"),
                "confidence_score": float(result.get("intent_confidence", 0)),
                "feedback": None, "feedback_comment": None,
                "model_used": CONFIG["llm_endpoint"],
                "token_count_in": None, "token_count_out": None,
                "latency_ms": latency_ms,
                "user_role": user_role,
                "created_at": datetime.now(),
            }]
            schema = StructType([
                StructField("conversation_id", StringType()), StructField("turn_number", IntegerType()),
                StructField("user_query", StringType()), StructField("intent_classified", StringType()),
                StructField("agent_used", StringType()), StructField("response_text", StringType()),
                StructField("retrieval_method", StringType()), StructField("sources_used", ArrayType(StringType())),
                StructField("sql_generated", StringType()), StructField("confidence_score", DoubleType()),
                StructField("feedback", StringType()), StructField("feedback_comment", StringType()),
                StructField("model_used", StringType()), StructField("token_count_in", IntegerType()),
                StructField("token_count_out", IntegerType()), StructField("latency_ms", IntegerType()),
                StructField("user_role", StringType()), StructField("created_at", TimestampType()),
            ])
            df = spark.createDataFrame(log, schema=schema)
            df.write.mode("append").saveAsTable(CONFIG["conversations_table"])
        except Exception as e:
            print(f"   ⚠️ Log error: {e}")


copilot = CopilotOrchestrator()
print(f"✅ Copilot Orchestrator ready — session: {copilot.conversation_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🧪 Tests

# COMMAND ----------

# Knowledge query
copilot.query("What are the business rules for Spirii EUH pipeline?")

# COMMAND ----------

# Data query → Genie
copilot.query("How many charging locations do we have by country?")

# COMMAND ----------

# Knowledge query
copilot.query("What deduplication logic does Driivz use in the raw layer?")

# COMMAND ----------

# Source overview
copilot.query("What is Enovos and what data does it provide?")

# COMMAND ----------

# Column-level question
copilot.query("How is the power_kw column derived in Spirii?")

# COMMAND ----------

# Data query → Genie
copilot.query("Show me the top 5 cities with the most charging stations")

# COMMAND ----------

# Vague question
copilot.query("How does data flow through the EV platform?")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check conversation log
# MAGIC SELECT turn_number, intent_classified, agent_used, latency_ms,
# MAGIC        LEFT(user_query, 60) as query,
# MAGIC        LEFT(response_text, 100) as answer_preview
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_conversations
# MAGIC ORDER BY created_at DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Activity 7 Complete
# MAGIC
# MAGIC **Built (all in one notebook):**
# MAGIC - Genie reporting views with column descriptions
# MAGIC - Intent Classifier (KNOWLEDGE / STRUCTURED / HYBRID)
# MAGIC - Knowledge Agent (structured-first + vector fallback + response composer)
# MAGIC - Genie Space API integration (async polling + table formatting)
# MAGIC - Python Orchestrator (routes, merges, logs)
# MAGIC - Conversation logging to Delta table
# MAGIC
# MAGIC **To do:**
# MAGIC 1. Update `genie_space_id` with your actual Space ID
# MAGIC 2. Adjust view column names if they don't match your tables
# MAGIC 3. Uncomment charger_sessions view if that table exists
# MAGIC
# MAGIC **Next → Activity 8: Chat UI (Gradio)**
