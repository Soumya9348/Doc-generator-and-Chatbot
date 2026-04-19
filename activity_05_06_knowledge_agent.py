# Databricks notebook source
# MAGIC %md
# MAGIC # 🧠 Activity 5 + 6: Knowledge Agent + Response Composer
# MAGIC
# MAGIC **Goal**: Build the Knowledge Agent with structured-first retrieval + vector fallback,
# MAGIC and the Response Composer that synthesizes answers with citations.
# MAGIC
# MAGIC **Flow**:
# MAGIC ```
# MAGIC User Query
# MAGIC     ↓
# MAGIC 1. Query Understanding (LLM → extract metadata)
# MAGIC     ↓
# MAGIC 2. Structured Retrieval (SQL WHERE on metadata columns)
# MAGIC     ↓
# MAGIC   ≥ 2 chunks found? → go to step 4
# MAGIC   < 2 chunks found? → step 3
# MAGIC     ↓
# MAGIC 3. Vector Search Fallback (with metadata pre-filters if available)
# MAGIC     ↓
# MAGIC 4. Response Composer (LLM → synthesize answer + citations)
# MAGIC     ↓
# MAGIC Final Answer
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration

# COMMAND ----------

import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import mlflow.deployments
from databricks.vector_search.client import VectorSearchClient

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
CONFIG = {
    # LLM endpoint — UPDATE with your actual Claude endpoint name
    "llm_endpoint": "PUT_YOUR_CLAUDE_SONNET_ENDPOINT_NAME_HERE",

    # Tables
    "knowledge_table": "`emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks",

    # Vector Search
    "vs_endpoint": "copilot-vs-endpoint",
    "vs_index": "`emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_index",

    # Embedding
    "embedding_endpoint": "databricks-bge-large-en",

    # Retrieval settings
    "structured_min_chunks": 2,   # If structured returns < this, fallback to vector
    "vector_top_k": 5,            # Number of chunks from vector search
    "max_chunks_for_synthesis": 3, # Cap chunks sent to Response Composer
    "confidence_threshold": 0.7,   # Min confidence to include a chunk

    # Sources
    "known_sources": ["driivz", "enovos", "spirii", "uberall"],
}

# Initialize clients
deploy_client = mlflow.deployments.get_deploy_client("databricks")
vsc = VectorSearchClient(disable_notice=True)

print("✅ Configuration loaded")
print(f"   LLM endpoint:  {CONFIG['llm_endpoint']}")
print(f"   VS index:      {CONFIG['vs_index']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Data Classes

# COMMAND ----------

@dataclass
class QueryMetadata:
    """Structured metadata extracted from a user query."""
    source_name: Optional[str] = None
    data_layer: Optional[str] = None
    section_type: Optional[str] = None
    tables_mentioned: list = field(default_factory=list)
    search_terms: list = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from knowledge base."""
    chunk_id: str = ""
    content: str = ""
    source_name: str = ""
    notebook_name: str = ""
    data_layer: str = ""
    section_header: str = ""
    retrieval_method: str = ""  # "structured" or "vector"
    relevance_score: float = 0.0


@dataclass
class AgentResponse:
    """Final response from the Knowledge Agent."""
    answer: str = ""
    citations: list = field(default_factory=list)
    retrieval_method: str = ""    # "structured", "vector", "hybrid"
    chunks_used: int = 0
    confidence: float = 0.0
    metadata_extracted: dict = field(default_factory=dict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Step 1: Query Understanding
# MAGIC
# MAGIC Parses the user's natural language → structured metadata for SQL lookup.
# MAGIC Uses LLM to extract source_name, data_layer, section_type, etc.

# COMMAND ----------

QUERY_UNDERSTANDING_PROMPT = """You are a query parser for the eMobility DataPlatform.

Given a user query, extract structured metadata to enable deterministic document retrieval.

Known source systems: driivz, enovos, spirii, uberall
Known data layers: landing, raw, euh
Known section types: path, notebook_purpose, data_layer, column_transformations, transformation_steps, business_rules, deduplication_logic, join_logic, error_handling, source_overview, table_details

Return ONLY valid JSON (no markdown, no explanation):
{
    "source_name": "<source system or null>",
    "data_layer": "<landing|raw|euh or null>",
    "section_type": "<section type or null>",
    "tables_mentioned": ["<table names if any>"],
    "search_terms": ["<key terms for search>"],
    "confidence": 0.XX
}

Examples:
- "How does Driivz EUH transform sessions?" → {"source_name": "driivz", "data_layer": "euh", "section_type": "transformation_steps", "tables_mentioned": ["sessions"], "search_terms": ["transform", "sessions"], "confidence": 0.95}
- "What business rules apply to Uberall data?" → {"source_name": "uberall", "data_layer": null, "section_type": "business_rules", "tables_mentioned": [], "search_terms": ["business rules"], "confidence": 0.85}
- "What deduplication is used in Spirii raw layer?" → {"source_name": "spirii", "data_layer": "raw", "section_type": "deduplication_logic", "tables_mentioned": [], "search_terms": ["deduplication"], "confidence": 0.9}
- "How is session_duration_seconds derived?" → {"source_name": null, "data_layer": "euh", "section_type": "column_transformations", "tables_mentioned": ["charger_session"], "search_terms": ["session_duration_seconds", "derived"], "confidence": 0.75}
- "Which country does Enovos operate in?" → {"source_name": "enovos", "data_layer": null, "section_type": "source_overview", "tables_mentioned": [], "search_terms": ["country", "operates"], "confidence": 0.9}
"""


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 500, temperature: float = 0) -> str:
    """Call the LLM endpoint and return the response text."""
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


def understand_query(user_query: str) -> QueryMetadata:
    """
    Step 1: Parse user query into structured metadata using LLM.
    This is the cheapest LLM call — just metadata extraction.
    """
    try:
        raw_response = call_llm(
            system_prompt=QUERY_UNDERSTANDING_PROMPT,
            user_message=user_query,
            max_tokens=200,
            temperature=0,
        )

        # Parse JSON from response (handle markdown code blocks)
        json_str = raw_response.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r'^```(?:json)?\n?', '', json_str)
            json_str = re.sub(r'\n?```$', '', json_str)

        parsed = json.loads(json_str)

        return QueryMetadata(
            source_name=parsed.get("source_name"),
            data_layer=parsed.get("data_layer"),
            section_type=parsed.get("section_type"),
            tables_mentioned=parsed.get("tables_mentioned", []),
            search_terms=parsed.get("search_terms", []),
            confidence=parsed.get("confidence", 0.5),
        )
    except Exception as e:
        print(f"   ⚠️ Query understanding error: {e}")
        # Fallback: basic keyword extraction
        return QueryMetadata(
            search_terms=user_query.lower().split(),
            confidence=0.3,
        )


# Quick test
print("✅ Query Understanding defined")
print("\n   Test: 'How does Driivz EUH transform sessions?'")
# Uncomment after setting the correct LLM endpoint:
# test_meta = understand_query("How does Driivz EUH transform sessions?")
# print(f"   Result: {test_meta}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🗄️ Step 2: Structured Retrieval
# MAGIC
# MAGIC SQL lookup against `copilot_knowledge_chunks` using extracted metadata.
# MAGIC This is the **zero-cost** retrieval path — no LLM, no embeddings, just SQL.

# COMMAND ----------

def structured_retrieval(metadata: QueryMetadata) -> list[RetrievedChunk]:
    """
    Step 2: SQL-based retrieval using extracted metadata.
    Builds dynamic WHERE clause from available metadata fields.
    Cost: $0 (just SQL).
    """
    conditions = []
    
    if metadata.source_name:
        conditions.append(f"source_name = '{metadata.source_name}'")
    
    if metadata.data_layer:
        conditions.append(f"data_layer = '{metadata.data_layer}'")
    
    if metadata.section_type:
        conditions.append(f"section_header = '{metadata.section_type}'")
    
    # Table mentions — check if any mentioned table appears in the tables_mentioned array
    if metadata.tables_mentioned:
        table_conditions = []
        for table in metadata.tables_mentioned:
            # Search in tables_mentioned array and also in content
            table_conditions.append(
                f"(array_contains(tables_mentioned, '{table}') OR LOWER(content) LIKE '%{table.lower()}%')"
            )
        if table_conditions:
            conditions.append(f"({' OR '.join(table_conditions)})")
    
    # Keyword search in content as fallback filter
    if metadata.search_terms and not conditions:
        # Only use keyword search if no other filters are available
        keyword_conditions = [f"LOWER(content) LIKE '%{term.lower()}%'" for term in metadata.search_terms[:3]]
        conditions.append(f"({' OR '.join(keyword_conditions)})")
    
    # Build query
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
        SELECT chunk_id, content, source_name, notebook_name, data_layer, 
               section_header, source_file_path
        FROM {CONFIG['knowledge_table']}
        WHERE {where_clause}
        ORDER BY 
            CASE WHEN section_header IN ('transformation_steps', 'column_transformations', 'business_rules') 
                 THEN 0 ELSE 1 END,
            chunk_index
        LIMIT 5
    """
    
    print(f"   📊 Structured SQL: WHERE {where_clause}")
    
    try:
        result_df = spark.sql(query)
        rows = result_df.collect()
        
        chunks = []
        for row in rows:
            chunks.append(RetrievedChunk(
                chunk_id=row.chunk_id,
                content=row.content,
                source_name=row.source_name or "",
                notebook_name=row.notebook_name or "",
                data_layer=row.data_layer or "",
                section_header=row.section_header or "",
                retrieval_method="structured",
                relevance_score=1.0,  # Exact match = highest relevance
            ))
        
        print(f"   📊 Structured retrieval: {len(chunks)} chunks found")
        return chunks
    
    except Exception as e:
        print(f"   ❌ Structured retrieval error: {e}")
        return []


print("✅ Structured Retrieval defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Step 3: Vector Search Fallback
# MAGIC
# MAGIC When structured retrieval returns < 2 chunks, fall back to semantic search.
# MAGIC Uses metadata pre-filters when available (hybrid approach).

# COMMAND ----------

def embed_query(query: str) -> list[float]:
    """Embed a query for vector search."""
    response = deploy_client.predict(
        endpoint=CONFIG["embedding_endpoint"],
        inputs={"input": [query]}
    )
    return response["data"][0]["embedding"]


def vector_search_fallback(user_query: str, metadata: QueryMetadata) -> list[RetrievedChunk]:
    """
    Step 3: Vector search with optional metadata pre-filters.
    Only called when structured retrieval returns insufficient results.
    """
    # Build metadata filters for vector search (narrow the search space)
    filters = {}
    if metadata.source_name:
        filters["source_name"] = metadata.source_name
    if metadata.data_layer:
        filters["data_layer"] = metadata.data_layer
    
    print(f"   🔍 Vector search: top-{CONFIG['vector_top_k']}, filters={filters or 'none'}")
    
    try:
        index = vsc.get_index(
            endpoint_name=CONFIG["vs_endpoint"],
            index_name=CONFIG["vs_index"],
        )
        
        search_kwargs = {
            "query_vector": embed_query(user_query),
            "columns": ["chunk_id", "content", "source_name", "notebook_name", 
                        "data_layer", "section_header"],
            "num_results": CONFIG["vector_top_k"],
        }
        
        if filters:
            search_kwargs["filters"] = filters
        
        results = index.similarity_search(**search_kwargs)
        
        chunks = []
        if results and "result" in results and results["result"]["data_array"]:
            for i, row in enumerate(results["result"]["data_array"]):
                # row order matches columns list above
                score = row[-1] if len(row) > 6 else (1.0 - i * 0.1)  # Similarity score or rank-based
                chunks.append(RetrievedChunk(
                    chunk_id=row[0] or "",
                    content=row[1] or "",
                    source_name=row[2] or "",
                    notebook_name=row[3] or "",
                    data_layer=row[4] or "",
                    section_header=row[5] or "",
                    retrieval_method="vector",
                    relevance_score=score if isinstance(score, float) else 0.8,
                ))
        
        print(f"   🔍 Vector search: {len(chunks)} chunks found")
        return chunks
    
    except Exception as e:
        print(f"   ❌ Vector search error: {e}")
        return []


print("✅ Vector Search Fallback defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ Step 4: Response Composer
# MAGIC
# MAGIC Takes retrieved chunks and synthesizes a final answer with citations.
# MAGIC Two paths:
# MAGIC - **Light synthesis**: Single high-confidence chunk → minimal LLM work
# MAGIC - **Full synthesis**: Multiple chunks → LLM merges and cites

# COMMAND ----------

RESPONSE_COMPOSER_PROMPT = """You are the Response Composer for the eMobility DataPlatform Copilot.

Your role: Take the retrieved documentation chunks and synthesize a clear, accurate answer.

Rules:
1. Answer the user's question directly — don't repeat the question back
2. Use ONLY the provided chunks — never make up information
3. Cite your sources using this format: [Source: notebook_name §section_name]
4. If the chunks don't fully answer the question, say what you know and what's missing
5. Format with markdown: headers, bullet points, code blocks as appropriate
6. Keep answers concise but complete
7. When discussing transformations, include relevant column names and logic
8. When multiple chunks cover the same topic, synthesize — don't just list them
9. If no chunks are relevant, say "I don't have documentation covering this topic."
"""


def compose_response(user_query: str, chunks: list[RetrievedChunk], retrieval_method: str) -> AgentResponse:
    """
    Step 4: Synthesize final answer from retrieved chunks.
    """
    if not chunks:
        return AgentResponse(
            answer="I don't have documentation covering this topic. Could you rephrase or ask about a specific source system (driivz, enovos, spirii, uberall)?",
            retrieval_method=retrieval_method,
            chunks_used=0,
            confidence=0.0,
        )
    
    # Deduplicate chunks by content
    seen = set()
    unique_chunks = []
    for c in chunks:
        content_key = c.content[:200]  # First 200 chars as dedup key
        if content_key not in seen:
            seen.add(content_key)
            unique_chunks.append(c)
    
    # Cap at max chunks
    top_chunks = unique_chunks[:CONFIG["max_chunks_for_synthesis"]]
    
    # Build context for LLM
    chunks_text = ""
    citations = []
    for i, chunk in enumerate(top_chunks):
        source_label = f"{chunk.notebook_name or chunk.source_name} §{chunk.section_header}"
        citations.append({
            "source": source_label,
            "retrieval_method": chunk.retrieval_method,
            "layer": chunk.data_layer,
        })
        chunks_text += f"\n--- Chunk {i+1} [Source: {source_label}] [Layer: {chunk.data_layer}] ---\n"
        chunks_text += chunk.content
        chunks_text += "\n"
    
    user_prompt = f"""User Question: {user_query}

Retrieved Documentation:
{chunks_text}

Synthesize a clear answer based on the chunks above. Include [Source: ...] citations."""
    
    try:
        answer = call_llm(
            system_prompt=RESPONSE_COMPOSER_PROMPT,
            user_message=user_prompt,
            max_tokens=800,
            temperature=0,
        )
        
        return AgentResponse(
            answer=answer,
            citations=citations,
            retrieval_method=retrieval_method,
            chunks_used=len(top_chunks),
            confidence=min(1.0, sum(c.relevance_score for c in top_chunks) / len(top_chunks)),
        )
    
    except Exception as e:
        print(f"   ❌ Response composition error: {e}")
        # Fallback: return raw chunks
        fallback_answer = f"Here are the relevant documentation sections:\n\n"
        for c in top_chunks:
            fallback_answer += f"**[{c.notebook_name} §{c.section_header}]** ({c.data_layer} layer)\n"
            fallback_answer += c.content[:500] + "\n\n"
        
        return AgentResponse(
            answer=fallback_answer,
            citations=citations,
            retrieval_method=retrieval_method,
            chunks_used=len(top_chunks),
            confidence=0.5,
        )


print("✅ Response Composer defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 Knowledge Agent — Full Pipeline

# COMMAND ----------

class KnowledgeAgent:
    """
    The Knowledge Agent: structured-first retrieval with vector fallback.
    
    Flow:
    1. Query Understanding → extract metadata (source, layer, section, tables)
    2. Structured Retrieval → SQL WHERE on metadata columns
    3. If < 2 chunks → Vector Search Fallback (with metadata pre-filters)
    4. Response Composer → LLM synthesizes answer with citations
    """
    
    def query(self, user_query: str, verbose: bool = True) -> AgentResponse:
        """
        Process a user query through the full retrieval pipeline.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"💬 Query: \"{user_query}\"")
            print(f"{'='*70}")
        
        # ─── Step 1: Query Understanding ───
        if verbose:
            print(f"\n🔍 Step 1: Query Understanding...")
        
        metadata = understand_query(user_query)
        
        if verbose:
            print(f"   source_name:  {metadata.source_name}")
            print(f"   data_layer:   {metadata.data_layer}")
            print(f"   section_type: {metadata.section_type}")
            print(f"   tables:       {metadata.tables_mentioned}")
            print(f"   search_terms: {metadata.search_terms}")
            print(f"   confidence:   {metadata.confidence}")
        
        # ─── Step 2: Structured Retrieval ───
        if verbose:
            print(f"\n🗄️ Step 2: Structured Retrieval...")
        
        structured_chunks = structured_retrieval(metadata)
        
        if verbose:
            for c in structured_chunks:
                print(f"   → [{c.retrieval_method}] {c.source_name}/{c.notebook_name} §{c.section_header}")
        
        # ─── Step 3: Vector Fallback (if needed) ───
        retrieval_method = "structured"
        all_chunks = structured_chunks
        
        if len(structured_chunks) < CONFIG["structured_min_chunks"]:
            if verbose:
                print(f"\n🔍 Step 3: Vector Search Fallback (structured returned {len(structured_chunks)} < {CONFIG['structured_min_chunks']})...")
            
            vector_chunks = vector_search_fallback(user_query, metadata)
            
            if verbose:
                for c in vector_chunks:
                    print(f"   → [{c.retrieval_method}] {c.source_name}/{c.notebook_name} §{c.section_header}")
            
            # Merge: structured first, then vector (avoid duplicates)
            seen_ids = {c.chunk_id for c in structured_chunks}
            for vc in vector_chunks:
                if vc.chunk_id not in seen_ids:
                    all_chunks.append(vc)
                    seen_ids.add(vc.chunk_id)
            
            retrieval_method = "hybrid" if structured_chunks else "vector"
        else:
            if verbose:
                print(f"\n✅ Step 3: Skipped — structured retrieval found enough chunks ({len(structured_chunks)} ≥ {CONFIG['structured_min_chunks']})")
        
        # ─── Step 4: Response Composer ───
        if verbose:
            print(f"\n✍️ Step 4: Response Composer ({len(all_chunks)} chunks → synthesis)...")
        
        response = compose_response(user_query, all_chunks, retrieval_method)
        response.metadata_extracted = {
            "source_name": metadata.source_name,
            "data_layer": metadata.data_layer,
            "section_type": metadata.section_type,
            "tables": metadata.tables_mentioned,
        }
        
        if verbose:
            print(f"\n{'─'*70}")
            print(f"📝 ANSWER ({response.retrieval_method.upper()} | {response.chunks_used} sources | conf: {response.confidence:.2f})")
            print(f"{'─'*70}")
            print(response.answer)
            print(f"\n📎 Citations:")
            for c in response.citations:
                print(f"   [{c['retrieval_method'].upper()}] {c['source']} (layer: {c['layer']})")
        
        return response


# Create the agent
agent = KnowledgeAgent()
print("✅ Knowledge Agent ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧪 Test: Run Knowledge Agent
# MAGIC
# MAGIC **IMPORTANT**: Before running, update `CONFIG["llm_endpoint"]` above with your actual Claude endpoint name.

# COMMAND ----------

# ─── Test with a single query first ───
response = agent.query("What are the business rules for Spirii EUH pipeline?")

# COMMAND ----------

# ─── Run full test suite ───
test_queries = [
    # Specific source + layer + section
    "How does the Driivz EUH pipeline transform the charger session table?",
    "What deduplication logic is used in Spirii raw layer?",
    "What are the business rules for Uberall?",
    
    # Source overview questions
    "Which country does Enovos operate in?",
    "What is Spirii used for?",
    
    # Column-level questions
    "How is the power_kw column derived in the Spirii EUH charger connector table?",
    
    # Cross-source questions (should check if it handles no source specified)
    "What join logic is used in the EUH layer across different sources?",
    
    # Error handling
    "How does the Driivz landing notebook handle errors?",
    
    # Vague question (should fallback to vector search)
    "How does data flow through the platform?",
    
    # Specific notebook question
    "What does the landing_etl_uberall_api notebook do?",
]

results = []
for q in test_queries:
    r = agent.query(q)
    results.append({
        "query": q,
        "method": r.retrieval_method,
        "chunks": r.chunks_used,
        "confidence": r.confidence,
        "answer_preview": r.answer[:150],
    })

# COMMAND ----------

# ─── Summary of all test results ───
print("\n" + "=" * 90)
print("📊 TEST RESULTS SUMMARY")
print("=" * 90)
print(f"\n{'Query':<60} {'Method':<12} {'Chunks':<8} {'Conf':<6}")
print(f"{'─'*60} {'─'*12} {'─'*8} {'─'*6}")

structured_count = 0
vector_count = 0
hybrid_count = 0

for r in results:
    method_emoji = {"structured": "🗄️", "vector": "🔍", "hybrid": "🔗"}.get(r["method"], "❓")
    print(f"{r['query'][:58]:<60} {method_emoji} {r['method']:<10} {r['chunks']:<8} {r['confidence']:.2f}")
    
    if r["method"] == "structured":
        structured_count += 1
    elif r["method"] == "vector":
        vector_count += 1
    else:
        hybrid_count += 1

print(f"\n📊 Retrieval method distribution:")
print(f"   🗄️ Structured: {structured_count}/{len(results)} ({structured_count/len(results)*100:.0f}%)")
print(f"   🔍 Vector:     {vector_count}/{len(results)} ({vector_count/len(results)*100:.0f}%)")
print(f"   🔗 Hybrid:     {hybrid_count}/{len(results)} ({hybrid_count/len(results)*100:.0f}%)")
print(f"\n   Target: structured ≥ 60% for specific queries ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Activity 5 + 6 Complete
# MAGIC
# MAGIC **Built:**
# MAGIC - `understand_query()` — LLM-based metadata extraction from natural language
# MAGIC - `structured_retrieval()` — SQL WHERE on source_name, data_layer, section_header, tables_mentioned
# MAGIC - `vector_search_fallback()` — semantic search with optional metadata pre-filters
# MAGIC - `compose_response()` — LLM synthesis with citations
# MAGIC - `KnowledgeAgent` class — orchestrates the full pipeline
# MAGIC
# MAGIC **Key behaviors:**
# MAGIC - Specific queries ("Spirii EUH business rules") → **structured retrieval** (fast, precise, $0 retrieval cost)
# MAGIC - Vague queries ("how does data flow?") → **vector search fallback** (semantic similarity)
# MAGIC - Partial metadata matches → **hybrid** (structured + vector merged)
# MAGIC
# MAGIC **Verify:**
# MAGIC 1. Are structured queries hitting the right chunks?
# MAGIC 2. Is vector fallback triggering for vague queries?
# MAGIC 3. Are answers accurate with proper citations?
# MAGIC
# MAGIC **Next → Activity 7: Supervisor Agent Setup**  
# MAGIC *(Activity 8: Chat UI can be built in parallel)*
