# Databricks notebook source
# MAGIC %md
# MAGIC # 🧭 Activity 7: Python Orchestrator (Supervisor Replacement)
# MAGIC
# MAGIC Since Supervisor Agent is a managed UI feature (not a Python SDK),
# MAGIC we build a **Python orchestrator** that does the same thing:
# MAGIC
# MAGIC 1. **Classify intent** using Claude (knowledge vs data query)
# MAGIC 2. **Route** to Knowledge Agent or Genie Space
# MAGIC 3. **Merge** results for hybrid queries
# MAGIC
# MAGIC ```
# MAGIC User Query → Intent Classifier → ┬→ Knowledge Agent → Answer
# MAGIC                                   ├→ Genie Space     → Answer
# MAGIC                                   └→ Both (HYBRID)   → Merged Answer
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration

# COMMAND ----------

import json
import re
import uuid
import time
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

    # Vector Search
    "vs_endpoint": "copilot-vs-endpoint",
    "vs_index": "emobility-uc-dev.sandbox-emobility.copilot_knowledge_index",

    # Genie Space — UPDATE with your Genie Space ID
    "genie_space_id": "PUT_YOUR_GENIE_SPACE_ID_HERE",

    # Retrieval settings
    "structured_min_chunks": 1,  # Lowered from 2 → even 1 exact match is good
    "vector_top_k": 5,
    "max_chunks_for_synthesis": 3,

    # Sources
    "known_sources": ["driivz", "enovos", "spirii", "uberall"],
}

deploy_client = mlflow.deployments.get_deploy_client("databricks")
vsc = VectorSearchClient(disable_notice=True)

print("✅ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 Intent Classifier
# MAGIC
# MAGIC Determines whether the query is a:
# MAGIC - **KNOWLEDGE_LOOKUP**: docs, architecture, transformations, business rules
# MAGIC - **STRUCTURED_QUERY**: metrics, counts, trends → route to Genie
# MAGIC - **HYBRID**: needs both data + explanation

# COMMAND ----------

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for the eMobility DataPlatform Copilot.

Classify the user's query into ONE of these categories:

KNOWLEDGE_LOOKUP — Questions about documentation, architecture, how pipelines work, transformation logic, business rules, column definitions, join logic, error handling, source system descriptions.
Examples: "How does Driivz EUH transform sessions?", "What are the business rules for Spirii?", "What deduplication logic is used?"

STRUCTURED_QUERY — Questions that need data/numbers from the database. Metrics, counts, totals, averages, trends, comparisons, "how many", "show me", "list all".
Examples: "How many active stations are there?", "What was the total revenue last month?", "Show sessions by country"

HYBRID — Questions that need BOTH data AND explanation. Typically "why" questions or questions that need context alongside numbers.
Examples: "Why did session count drop last week?", "What's the trend and how does the pipeline handle it?"

Return ONLY valid JSON (no markdown, no explanation):
{"intent": "KNOWLEDGE_LOOKUP|STRUCTURED_QUERY|HYBRID", "confidence": 0.XX, "reasoning": "brief reason"}
"""


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 500, temperature: float = 0) -> str:
    """Call Claude endpoint."""
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


def classify_intent(user_query: str) -> dict:
    """
    Classify the user's intent: KNOWLEDGE_LOOKUP, STRUCTURED_QUERY, or HYBRID.
    """
    try:
        raw = call_llm(INTENT_CLASSIFIER_PROMPT, user_query, max_tokens=100, temperature=0)
        json_str = raw.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r'^```(?:json)?\n?', '', json_str)
            json_str = re.sub(r'\n?```$', '', json_str)
        parsed = json.loads(json_str)
        return {
            "intent": parsed.get("intent", "KNOWLEDGE_LOOKUP"),
            "confidence": parsed.get("confidence", 0.5),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        print(f"   ⚠️ Intent classification error: {e}")
        return {"intent": "KNOWLEDGE_LOOKUP", "confidence": 0.5, "reasoning": "fallback"}


# Test
print("✅ Intent Classifier defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Knowledge Agent (from Activity 5+6)
# MAGIC
# MAGIC Copied here so this notebook is self-contained.

# COMMAND ----------

# ─────────────────────────────────────────
# Query Understanding
# ─────────────────────────────────────────
QUERY_UNDERSTANDING_PROMPT = """You are a query parser for the eMobility DataPlatform.

Given a user query, extract structured metadata to enable deterministic document retrieval.

Known source systems: driivz, enovos, spirii, uberall
Known data layers: landing, raw, euh
Known section types: path, notebook_purpose, data_layer, column_transformations, transformation_steps, business_rules, deduplication_logic, join_logic, error_handling, source_overview, table_details

Return ONLY valid JSON:
{"source_name": "<or null>", "data_layer": "<landing|raw|euh or null>", "section_type": "<or null>", "tables_mentioned": [], "search_terms": [], "confidence": 0.XX}
"""


def understand_query(user_query: str) -> dict:
    try:
        raw = call_llm(QUERY_UNDERSTANDING_PROMPT, user_query, max_tokens=200, temperature=0)
        json_str = raw.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r'^```(?:json)?\n?', '', json_str)
            json_str = re.sub(r'\n?```$', '', json_str)
        return json.loads(json_str)
    except:
        return {"source_name": None, "data_layer": None, "section_type": None,
                "tables_mentioned": [], "search_terms": user_query.lower().split(), "confidence": 0.3}


# ─────────────────────────────────────────
# Structured Retrieval
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Vector Search Fallback
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Response Composer
# ─────────────────────────────────────────
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


def compose_response(user_query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"answer": "I don't have documentation covering this topic. Try asking about driivz, enovos, spirii, or uberall.",
                "citations": [], "method": "none"}

    # Dedup and cap
    seen = set()
    unique = []
    for c in chunks:
        key = c["content"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    top = unique[:CONFIG["max_chunks_for_synthesis"]]

    # Build context
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
        answer = f"Error generating response: {e}\n\nRaw chunks:\n"
        for c in top:
            answer += f"\n**[{c['notebook_name']} §{c['section_header']}]**\n{c['content'][:300]}\n"

    return {"answer": answer, "citations": citations, "method": top[0]["method"] if top else "none"}


def knowledge_agent(user_query: str) -> dict:
    """Full Knowledge Agent pipeline: understand → structured → [vector fallback] → compose."""
    metadata = understand_query(user_query)
    structured = structured_retrieval(metadata)

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

    result = compose_response(user_query, all_chunks)
    result["retrieval_method"] = method
    result["metadata"] = metadata
    return result


print("✅ Knowledge Agent defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Genie Space Integration
# MAGIC
# MAGIC For structured data queries (metrics, counts, trends).
# MAGIC Uses the Databricks Genie API to query your reporting views.

# COMMAND ----------

def genie_query(user_query: str) -> dict:
    """
    Route a data question to Genie Space.
    Uses the Databricks SDK Genie API.
    """
    space_id = CONFIG["genie_space_id"]

    if space_id == "PUT_YOUR_GENIE_SPACE_ID_HERE":
        return {
            "answer": "⚠️ **Genie Space not configured.** Please update `CONFIG['genie_space_id']` with your Genie Space ID.\n\nYou can find it in the Databricks UI: **Genie → Your Space → URL contains the space ID.**",
            "citations": [{"source": "Genie Space", "method": "genie", "layer": "reporting"}],
            "retrieval_method": "genie",
            "sql_generated": None,
        }

    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        # Start a Genie conversation
        response = w.genie.start_conversation(
            space_id=space_id,
            content=user_query,
        )

        # The Genie API is async — poll for results
        conversation_id = response.conversation_id
        message_id = response.message_id

        # Poll for completion (Genie processes the query)
        max_wait = 60
        waited = 0
        result = None

        while waited < max_wait:
            try:
                msg = w.genie.get_message(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                )
                status = msg.status
                if status == "COMPLETED":
                    result = msg
                    break
                elif status in ("FAILED", "CANCELLED"):
                    return {
                        "answer": f"Genie query failed with status: {status}",
                        "citations": [{"source": "Genie Space", "method": "genie", "layer": "reporting"}],
                        "retrieval_method": "genie",
                        "sql_generated": None,
                    }
                time.sleep(3)
                waited += 3
            except Exception:
                time.sleep(3)
                waited += 3

        if result is None:
            return {
                "answer": "Genie query timed out. Please try a simpler question.",
                "citations": [], "retrieval_method": "genie", "sql_generated": None,
            }

        # Extract answer from Genie response
        # The structure may vary — adapt based on actual API response
        answer_text = ""
        sql_generated = None

        for attachment in (result.attachments or []):
            if attachment.text:
                answer_text += attachment.text.content + "\n"
            if attachment.query:
                sql_generated = attachment.query.query
                # If there are query results, format them
                if attachment.query.result:
                    answer_text += "\n**Query Results:**\n"
                    # Format as table
                    columns = [col.name for col in (attachment.query.result.columns or [])]
                    if columns:
                        answer_text += "| " + " | ".join(columns) + " |\n"
                        answer_text += "| " + " | ".join(["---"] * len(columns)) + " |\n"
                        for row in (attachment.query.result.data_array or [])[:20]:
                            answer_text += "| " + " | ".join(str(v) for v in row) + " |\n"

        if not answer_text:
            answer_text = "Genie processed your query but returned no text response."

        return {
            "answer": answer_text.strip(),
            "citations": [{"source": "Genie Space", "method": "genie", "layer": "reporting"}],
            "retrieval_method": "genie",
            "sql_generated": sql_generated,
        }

    except ImportError:
        return {
            "answer": "⚠️ `databricks-sdk` not available. Install it or update your cluster.",
            "citations": [], "retrieval_method": "genie", "sql_generated": None,
        }
    except Exception as e:
        return {
            "answer": f"Genie query error: {str(e)}",
            "citations": [], "retrieval_method": "genie", "sql_generated": None,
        }


print("✅ Genie Space integration defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧭 The Orchestrator
# MAGIC
# MAGIC Routes queries to the correct agent based on intent classification.

# COMMAND ----------

class CopilotOrchestrator:
    """
    The main orchestrator — replaces Supervisor Agent with Python logic.
    
    Flow:
    1. Classify intent (Claude)
    2. Route to Knowledge Agent or Genie Space
    3. For HYBRID, call both and merge
    4. Log conversation turn
    """
    
    def __init__(self):
        self.conversation_id = str(uuid.uuid4())
        self.turn_number = 0
        self.history = []  # Last N turns for context
    
    def query(self, user_query: str, user_role: str = "Engineer") -> dict:
        """Process a user query through the full pipeline."""
        self.turn_number += 1
        start_time = time.time()
        
        # ─── Step 1: Intent Classification ───
        intent_result = classify_intent(user_query)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        
        print(f"\n{'='*70}")
        print(f"💬 [{self.turn_number}] \"{user_query}\"")
        print(f"   🎯 Intent: {intent} (conf: {confidence:.2f}) — {intent_result['reasoning']}")
        
        # ─── Step 2: Route to agent(s) ───
        result = None
        
        if intent == "KNOWLEDGE_LOOKUP":
            print(f"   📚 Routing → Knowledge Agent")
            result = knowledge_agent(user_query)
            
        elif intent == "STRUCTURED_QUERY":
            print(f"   📊 Routing → Genie Space")
            result = genie_query(user_query)
            
        elif intent == "HYBRID":
            print(f"   🔗 Routing → Knowledge Agent + Genie Space")
            knowledge_result = knowledge_agent(user_query)
            genie_result = genie_query(user_query)
            
            # Merge: knowledge answer first, then data
            merged_answer = ""
            if knowledge_result.get("answer"):
                merged_answer += "### 📚 From Documentation\n\n"
                merged_answer += knowledge_result["answer"] + "\n\n"
            if genie_result.get("answer") and "not configured" not in genie_result["answer"]:
                merged_answer += "### 📊 From Data\n\n"
                merged_answer += genie_result["answer"]
            
            result = {
                "answer": merged_answer or knowledge_result.get("answer", "No results."),
                "citations": knowledge_result.get("citations", []) + genie_result.get("citations", []),
                "retrieval_method": "hybrid",
                "sql_generated": genie_result.get("sql_generated"),
            }
        else:
            # Default to knowledge
            result = knowledge_agent(user_query)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # ─── Add metadata ───
        result["intent"] = intent
        result["intent_confidence"] = confidence
        result["latency_ms"] = latency_ms
        result["turn_number"] = self.turn_number
        result["conversation_id"] = self.conversation_id
        
        # ─── Store in history ───
        self.history.append({
            "turn": self.turn_number,
            "query": user_query,
            "intent": intent,
            "answer_preview": result["answer"][:200],
        })
        # Keep last 5 turns
        if len(self.history) > 5:
            self.history = self.history[-5:]
        
        # ─── Log to conversations table ───
        self._log_conversation(user_query, result, user_role, latency_ms)
        
        # ─── Display ───
        print(f"\n{'─'*70}")
        print(f"📝 Answer ({result['retrieval_method'].upper()} | ⏱️ {latency_ms}ms)")
        print(f"{'─'*70}")
        print(result["answer"][:1000])
        if len(result["answer"]) > 1000:
            print(f"\n... [{len(result['answer']) - 1000} more chars]")
        
        if result.get("citations"):
            print(f"\n📎 Citations:")
            for c in result["citations"]:
                print(f"   [{c['method'].upper()}] {c['source']}")
        
        if result.get("sql_generated"):
            print(f"\n💾 SQL Generated:\n{result['sql_generated']}")
        
        return result
    
    def _log_conversation(self, user_query: str, result: dict, user_role: str, latency_ms: int):
        """Log this conversation turn to Delta table."""
        try:
            log_data = [{
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
                "feedback": None,
                "feedback_comment": None,
                "model_used": CONFIG["llm_endpoint"],
                "token_count_in": None,
                "token_count_out": None,
                "latency_ms": latency_ms,
                "user_role": user_role,
                "created_at": datetime.now(),
            }]
            
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, ArrayType
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
            df = spark.createDataFrame(log_data, schema=schema)
            df.write.mode("append").saveAsTable(CONFIG["conversations_table"])
        except Exception as e:
            print(f"   ⚠️ Logging error (non-fatal): {e}")


# Create the orchestrator
copilot = CopilotOrchestrator()
print("✅ Copilot Orchestrator ready")
print(f"   Session: {copilot.conversation_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧪 Test the Orchestrator

# COMMAND ----------

# Test 1: Knowledge query
copilot.query("What are the business rules for Spirii EUH pipeline?")

# COMMAND ----------

# Test 2: Should route to Genie (or show placeholder if not configured)
copilot.query("How many charging sessions happened last month?")

# COMMAND ----------

# Test 3: Knowledge query
copilot.query("What deduplication logic does Driivz use in the raw layer?")

# COMMAND ----------

# Test 4: Source overview
copilot.query("What is Enovos and what data does it provide?")

# COMMAND ----------

# Test 5: Column-level question
copilot.query("How is the power_kw column derived in Spirii?")

# COMMAND ----------

# Test 6: Vague question
copilot.query("How does data flow through the EV platform?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Check Conversation Log

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT turn_number, intent_classified, agent_used, latency_ms,
# MAGIC        LEFT(user_query, 60) as query,
# MAGIC        LEFT(response_text, 100) as answer_preview
# MAGIC FROM emobility-uc-dev.`sandbox-emobility`.copilot_conversations
# MAGIC ORDER BY created_at DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Activity 7 Complete
# MAGIC
# MAGIC **Built:**
# MAGIC - **Intent Classifier**: Claude-based routing (KNOWLEDGE_LOOKUP / STRUCTURED_QUERY / HYBRID)
# MAGIC - **Python Orchestrator**: Routes to Knowledge Agent or Genie Space
# MAGIC - **Genie Integration**: API stub (update with your Genie Space ID)
# MAGIC - **Conversation Logging**: Every query logged to `copilot_conversations` table
# MAGIC - **Session History**: Keeps last 5 turns for future multi-turn context
# MAGIC
# MAGIC **To complete Genie integration:**
# MAGIC 1. Get your Genie Space ID from the Databricks UI (Genie → your space → URL)
# MAGIC 2. Update `CONFIG["genie_space_id"]`
# MAGIC 3. Re-run a STRUCTURED_QUERY test
# MAGIC
# MAGIC **Next → Activity 8: Chat UI (Gradio)**
