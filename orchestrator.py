"""
eMobility Copilot Orchestrator — Refactored for Databricks App deployment.
Standalone Python module (no notebook dependencies).
Uses databricks-sql-connector instead of SparkSession.
"""

import json
import re
import os
import uuid
import time
from datetime import datetime

import mlflow.deployments

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

CONFIG = {
    "llm_endpoint": "databricks-claude-sonnet-4-6",
    "embedding_endpoint": "databricks-gte-large-en",
    "knowledge_table": "emobility-uc-dev.sandbox-emobility.copilot_knowledge_chunks",
    "conversations_table": "emobility-uc-dev.sandbox-emobility.copilot_conversations",
    "vs_endpoint": "copilot-vs-endpoint",
    "vs_index": "emobility-uc-dev.sandbox-emobility.copilot_knowledge_index",
    "genie_space_id": os.environ.get("GENIE_SPACE_ID", "PUT_YOUR_GENIE_SPACE_ID_HERE"),
    "structured_min_chunks": 1,
    "vector_top_k": 5,
    "max_chunks_for_synthesis": 3,
    "known_sources": ["driivz", "enovos", "spirii", "uberall"],
}

# ─────────────────────────────────────────────────────────────────
# Clients (initialized lazily)
# ─────────────────────────────────────────────────────────────────

_deploy_client = None
_sql_connection = None
_vsc = None


def get_deploy_client():
    global _deploy_client
    if _deploy_client is None:
        _deploy_client = mlflow.deployments.get_deploy_client("databricks")
    return _deploy_client


def get_sql_connection():
    """Get a databricks-sql-connector connection for Delta table queries."""
    global _sql_connection
    if _sql_connection is None:
        from databricks import sql as dbsql
        _sql_connection = dbsql.connect(
            server_hostname=os.environ.get("DATABRICKS_SERVER_HOSTNAME", ""),
            http_path=os.environ.get("DATABRICKS_HTTP_PATH", ""),
            access_token=os.environ.get("DATABRICKS_TOKEN", ""),
        )
    return _sql_connection


def get_vector_client():
    global _vsc
    if _vsc is None:
        from databricks.vector_search.client import VectorSearchClient
        _vsc = VectorSearchClient(disable_notice=True)
    return _vsc


def run_sql(query: str) -> list[dict]:
    """Execute SQL and return list of dicts."""
    conn = get_sql_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        print(f"SQL error: {e}")
        return []
    finally:
        cursor.close()


# ─────────────────────────────────────────────────────────────────
# LLM Helper
# ─────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_message: str, max_tokens: int = 500, temperature: float = 0) -> str:
    """Call Claude endpoint and return response text."""
    response = get_deploy_client().predict(
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


# ─────────────────────────────────────────────────────────────────
# Intent Classifier
# ─────────────────────────────────────────────────────────────────

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for the eMobility DataPlatform Copilot.

Classify the user's query into ONE of these categories:

KNOWLEDGE_LOOKUP — Questions about documentation, architecture, how pipelines work, transformation logic, business rules, column definitions, join logic, error handling, source system descriptions.
Examples: "How does Driivz EUH transform sessions?", "What are the business rules for Spirii?", "What deduplication logic is used?"

STRUCTURED_QUERY — Questions that need data/numbers from the database. Metrics, counts, totals, averages, trends, comparisons, "how many", "show me", "list all".
Examples: "How many active stations are there?", "What was the total revenue last month?", "Show sessions by country"

HYBRID — Questions that need BOTH data AND explanation.
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
        print(f"Intent classification error: {e}")
        return {"intent": "KNOWLEDGE_LOOKUP", "confidence": 0.5, "reasoning": "fallback"}


# ─────────────────────────────────────────────────────────────────
# Knowledge Agent
# ─────────────────────────────────────────────────────────────────

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


def understand_query(user_query: str) -> dict:
    try:
        raw = call_llm(QUERY_UNDERSTANDING_PROMPT, user_query, max_tokens=200, temperature=0)
        return parse_llm_json(raw)
    except:
        return {"source_name": None, "data_layer": None, "section_type": None,
                "tables_mentioned": [], "search_terms": user_query.lower().split(), "confidence": 0.3}


def structured_retrieval(metadata: dict) -> list[dict]:
    """Query Delta table with deterministic SQL filters."""
    conditions = []
    if metadata.get("source_name"):
        conditions.append(f"source_name = '{metadata['source_name']}'")
    if metadata.get("data_layer"):
        conditions.append(f"data_layer = '{metadata['data_layer']}'")
    if metadata.get("section_type"):
        conditions.append(f"section_header = '{metadata['section_type']}'")
    if metadata.get("tables_mentioned"):
        tbl_conds = [f"LOWER(content) LIKE '%{t.lower()}%'" for t in metadata["tables_mentioned"]]
        conditions.append(f"({' OR '.join(tbl_conds)})")
    if metadata.get("search_terms") and not conditions:
        kw_conds = [f"LOWER(content) LIKE '%{t.lower()}%'" for t in metadata["search_terms"][:3]]
        conditions.append(f"({' OR '.join(kw_conds)})")

    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"""SELECT chunk_id, content, source_name, notebook_name, data_layer, section_header
                FROM {CONFIG['knowledge_table']} WHERE {where}
                ORDER BY chunk_index LIMIT 5"""
    rows = run_sql(query)
    return [{"chunk_id": r.get("chunk_id", ""), "content": r.get("content", ""),
             "source_name": r.get("source_name", ""), "notebook_name": r.get("notebook_name", ""),
             "data_layer": r.get("data_layer", ""), "section_header": r.get("section_header", ""),
             "method": "structured", "score": 1.0}
            for r in rows]


def embed_query(query: str) -> list[float]:
    response = get_deploy_client().predict(
        endpoint=CONFIG["embedding_endpoint"], inputs={"input": [query]}
    )
    return response["data"][0]["embedding"]


def vector_search(user_query: str, metadata: dict) -> list[dict]:
    """Fallback: similarity search via Vector Search index."""
    filters = {}
    if metadata.get("source_name"):
        filters["source_name"] = metadata["source_name"]
    if metadata.get("data_layer"):
        filters["data_layer"] = metadata["data_layer"]

    try:
        vsc = get_vector_client()
        index = vsc.get_index(endpoint_name=CONFIG["vs_endpoint"], index_name=CONFIG["vs_index"])
        kwargs = {
            "query_vector": embed_query(user_query),
            "columns": ["chunk_id", "content", "source_name", "notebook_name", "data_layer", "section_header"],
            "num_results": CONFIG["vector_top_k"],
        }
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
        print(f"Vector search error: {e}")
        return []


def compose_response(user_query: str, chunks: list[dict]) -> dict:
    """Synthesize answer from retrieved chunks via LLM."""
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


def knowledge_agent(user_query: str) -> dict:
    """Full pipeline: understand → structured → [vector fallback] → compose."""
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


# ─────────────────────────────────────────────────────────────────
# Genie Space Integration
# ─────────────────────────────────────────────────────────────────

def genie_query(user_query: str) -> dict:
    """Route a data question to Genie Space via the Databricks SDK."""
    space_id = CONFIG["genie_space_id"]

    if space_id == "PUT_YOUR_GENIE_SPACE_ID_HERE":
        return {
            "answer": "⚠️ **Genie Space not configured.** Update `GENIE_SPACE_ID` environment variable.",
            "citations": [{"source": "Genie Space", "method": "genie", "layer": "reporting"}],
            "retrieval_method": "genie", "sql_generated": None, "chart_data": None,
        }

    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        response = w.genie.start_conversation(space_id=space_id, content=user_query)
        conversation_id = response.conversation_id
        message_id = response.message_id

        max_wait, waited = 120, 0
        result = None
        while waited < max_wait:
            msg = w.genie.get_message(space_id=space_id, conversation_id=conversation_id, message_id=message_id)
            status = msg.status.value if hasattr(msg.status, 'value') else str(msg.status)
            if status in ("COMPLETED", "COMPLETED_WITH_ERROR"):
                result = msg
                break
            elif status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
                return {"answer": f"Genie query failed: {status}", "citations": [],
                        "retrieval_method": "genie", "sql_generated": None, "chart_data": None}
            time.sleep(3)
            waited += 3

        if result is None:
            return {"answer": "Genie query timed out.", "citations": [],
                    "retrieval_method": "genie", "sql_generated": None, "chart_data": None}

        answer_text = ""
        sql_generated = None
        chart_data = None

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
                        # Build markdown table
                        answer_text += "\n| " + " | ".join(columns) + " |\n"
                        answer_text += "| " + " | ".join(["---"] * len(columns)) + " |\n"
                        for row in data[:20]:
                            answer_text += "| " + " | ".join(str(v or "") for v in row) + " |\n"

                        # Build chart_data for frontend
                        chart_data = {"columns": columns, "rows": [list(row) for row in data[:50]]}
                        # Auto-detect x/y columns
                        num_cols = []
                        cat_cols = []
                        for i, col in enumerate(columns):
                            sample_vals = [row[i] for row in data[:5] if row[i] is not None]
                            try:
                                [float(v) for v in sample_vals]
                                num_cols.append(col)
                            except (ValueError, TypeError):
                                cat_cols.append(col)
                        if cat_cols and num_cols:
                            chart_data["x_col"] = cat_cols[0]
                            chart_data["y_col"] = num_cols[0]
                            chart_data["suggested_type"] = "pie" if len(data) <= 6 else "bar"

        return {
            "answer": answer_text.strip() or "Query completed but no text returned.",
            "citations": [{"source": "Genie Space", "method": "genie", "layer": "reporting"}],
            "retrieval_method": "genie",
            "sql_generated": sql_generated,
            "chart_data": chart_data,
        }

    except ImportError:
        return {"answer": "⚠️ `databricks-sdk` not available.", "citations": [],
                "retrieval_method": "genie", "sql_generated": None, "chart_data": None}
    except Exception as e:
        return {"answer": f"Genie error: {str(e)}", "citations": [],
                "retrieval_method": "genie", "sql_generated": None, "chart_data": None}


# ─────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────

class CopilotOrchestrator:
    """Main orchestrator — classifies intent, routes to agents, logs conversations."""

    def __init__(self):
        self.conversation_id = str(uuid.uuid4())
        self.turn_number = 0
        self.history = []

    def query(self, user_query: str, user_role: str = "Engineer") -> dict:
        """Process a user query through the full pipeline."""
        self.turn_number += 1
        start_time = time.time()

        # Step 1: Intent
        intent_result = classify_intent(user_query)
        intent = intent_result["intent"]

        # Step 2: Route
        result = None
        if intent == "KNOWLEDGE_LOOKUP":
            result = knowledge_agent(user_query)
        elif intent == "STRUCTURED_QUERY":
            result = genie_query(user_query)
        elif intent == "HYBRID":
            k_result = knowledge_agent(user_query)
            g_result = genie_query(user_query)
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
                "chart_data": g_result.get("chart_data"),
            }
        else:
            result = knowledge_agent(user_query)

        latency_ms = int((time.time() - start_time) * 1000)

        # Enrich
        result["intent"] = intent
        result["intent_confidence"] = intent_result["confidence"]
        result["latency_ms"] = latency_ms
        result["turn_number"] = self.turn_number
        result["conversation_id"] = self.conversation_id
        if "chart_data" not in result:
            result["chart_data"] = None

        # History
        self.history.append({"turn": self.turn_number, "query": user_query, "intent": intent})
        if len(self.history) > 5:
            self.history = self.history[-5:]

        # Log
        self._log_conversation(user_query, result, user_role, latency_ms)

        return result

    def _log_conversation(self, user_query, result, user_role, latency_ms):
        """Log turn to conversations table via SQL INSERT."""
        try:
            answer_escaped = result.get("answer", "")[:5000].replace("'", "''")
            query_escaped = user_query.replace("'", "''")
            sources = json.dumps([c["source"] for c in result.get("citations", [])])
            sql_gen = (result.get("sql_generated") or "").replace("'", "''")

            insert_sql = f"""
                INSERT INTO {CONFIG['conversations_table']}
                (conversation_id, turn_number, user_query, intent_classified, agent_used,
                 response_text, retrieval_method, sql_generated, confidence_score,
                 model_used, latency_ms, user_role, created_at)
                VALUES (
                    '{self.conversation_id}', {self.turn_number}, '{query_escaped}',
                    '{result.get("intent", "")}', '{result.get("retrieval_method", "")}',
                    '{answer_escaped}', '{result.get("retrieval_method", "")}',
                    '{sql_gen}', {float(result.get("intent_confidence", 0))},
                    '{CONFIG["llm_endpoint"]}', {latency_ms}, '{user_role}',
                    current_timestamp()
                )
            """
            run_sql(insert_sql)
        except Exception as e:
            print(f"Log error: {e}")
