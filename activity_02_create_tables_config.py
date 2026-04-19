# Databricks notebook source
# MAGIC %md
# MAGIC # 🗄️ Activity 2: Create Copilot Tables + Config
# MAGIC
# MAGIC **Goal**: Create the catalog schema, Delta tables, and Volume structure needed for the copilot.
# MAGIC
# MAGIC **Tasks**:
# MAGIC - 2.1 — Verify/create schema: `emobility-uc-dev`.`sandbox-emobility`
# MAGIC - 2.2 — Create `copilot_knowledge_chunks` table
# MAGIC - 2.3 — Create `copilot_conversations` table
# MAGIC - 2.4 — Create Volume structure for config files (agent prompts, routing rules)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 — Verify Schema Exists

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify the schema exists (it should already exist as sandbox-emobility)
# MAGIC DESCRIBE SCHEMA `emobility-uc-dev`.`sandbox-emobility`;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 — Create `copilot_knowledge_chunks` Table
# MAGIC
# MAGIC This is the **core table** for the RAG system. Every doc chunk + KT chunk goes here,
# MAGIC enriched with metadata that powers structured-first retrieval.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks (
# MAGIC     -- Identity
# MAGIC     chunk_id            STRING          COMMENT 'UUID for each chunk',
# MAGIC     content             STRING          COMMENT 'The actual text content of this chunk',
# MAGIC     content_hash        STRING          COMMENT 'SHA-256 hash of content — for deduplication on re-ingestion',
# MAGIC
# MAGIC     -- Embedding
# MAGIC     embedding           ARRAY<FLOAT>    COMMENT '1024-dim vector embedding (BGE-large)',
# MAGIC
# MAGIC     -- Structured metadata (powers structured-first retrieval)
# MAGIC     source_type         STRING          COMMENT 'doc_generator | kt_transcript | runbook',
# MAGIC     source_name         STRING          COMMENT 'Source system: driivz | ecomovement | cxm | newmotion | greenlots',
# MAGIC     document_type       STRING          COMMENT 'business_overview | notebook_doc | kt_transcript',
# MAGIC     data_layer          STRING          COMMENT 'landing | raw | euh | NULL (for business overviews)',
# MAGIC     notebook_name       STRING          COMMENT 'Name of the notebook this chunk belongs to (e.g. landing_etl_driivz_api)',
# MAGIC
# MAGIC     -- Content metadata
# MAGIC     section_header      STRING          COMMENT 'Section name: purpose | data_layer | column_transformations | transformation_steps | business_rules | merging_joining',
# MAGIC     tables_mentioned    ARRAY<STRING>   COMMENT 'Table names referenced in this chunk (catalog.schema.table)',
# MAGIC     keywords            ARRAY<STRING>   COMMENT 'Top-10 extracted keywords for hybrid search',
# MAGIC
# MAGIC     -- Source tracking
# MAGIC     source_file_path    STRING          COMMENT 'Full Volume path of the source file',
# MAGIC     chunk_index         INT             COMMENT 'Position of this chunk within the source document (0-based)',
# MAGIC     total_chunks        INT             COMMENT 'Total number of chunks from this source document',
# MAGIC
# MAGIC     -- Timestamps
# MAGIC     created_at          TIMESTAMP       COMMENT 'When this chunk was ingested',
# MAGIC     updated_at          TIMESTAMP       COMMENT 'When this chunk was last updated'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Knowledge chunks for DataPlatform Copilot RAG — enriched with structured metadata for deterministic retrieval'
# MAGIC TBLPROPERTIES (
# MAGIC     'delta.enableChangeDataFeed' = 'true',
# MAGIC     'delta.autoOptimize.optimizeWrite' = 'true',
# MAGIC     'delta.autoOptimize.autoCompact' = 'true'
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 — Create `copilot_conversations` Table
# MAGIC
# MAGIC Stores conversation history for multi-turn context and analytics.
# MAGIC Using Delta table for the demo (Lakebase decision deferred to Activity 7).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS `emobility-uc-dev`.`sandbox-emobility`.copilot_conversations (
# MAGIC     -- Session
# MAGIC     conversation_id     STRING          COMMENT 'Unique session ID',
# MAGIC     turn_number         INT             COMMENT 'Turn number within this conversation (1-based)',
# MAGIC
# MAGIC     -- Query & Response
# MAGIC     user_query          STRING          COMMENT 'The user''s question',
# MAGIC     intent_classified   STRING          COMMENT 'STRUCTURED_QUERY | KNOWLEDGE_LOOKUP | PIPELINE_STATUS | SQL_REVIEW | HYBRID',
# MAGIC     agent_used          STRING          COMMENT 'Which agent handled this: knowledge | genie | pipeline | sql_advisor',
# MAGIC     response_text       STRING          COMMENT 'The full response text returned to the user',
# MAGIC     retrieval_method    STRING          COMMENT 'structured | vector | hybrid | genie',
# MAGIC
# MAGIC     -- Sources
# MAGIC     sources_used        ARRAY<STRING>   COMMENT 'Source references cited in the response',
# MAGIC     sql_generated       STRING          COMMENT 'SQL generated by Genie (NULL if not applicable)',
# MAGIC
# MAGIC     -- Quality metrics
# MAGIC     confidence_score    DOUBLE          COMMENT 'Response Composer confidence (0.0 to 1.0)',
# MAGIC     feedback            STRING          COMMENT 'positive | negative | NULL',
# MAGIC     feedback_comment    STRING          COMMENT 'Optional user feedback text',
# MAGIC
# MAGIC     -- Cost tracking
# MAGIC     model_used          STRING          COMMENT 'Which LLM model was used for final synthesis',
# MAGIC     token_count_in      INT             COMMENT 'Total input tokens consumed',
# MAGIC     token_count_out     INT             COMMENT 'Total output tokens consumed',
# MAGIC     latency_ms          INT             COMMENT 'End-to-end response latency in milliseconds',
# MAGIC
# MAGIC     -- User context
# MAGIC     user_role           STRING          COMMENT 'BA | Engineer | Leadership',
# MAGIC
# MAGIC     -- Timestamps
# MAGIC     created_at          TIMESTAMP       COMMENT 'When this turn was created'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Conversation history for DataPlatform Copilot — multi-turn context and analytics'
# MAGIC TBLPROPERTIES (
# MAGIC     'delta.enableChangeDataFeed' = 'true',
# MAGIC     'delta.autoOptimize.optimizeWrite' = 'true',
# MAGIC     'delta.autoOptimize.autoCompact' = 'true'
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 — Create Volume Structure for Config Files

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a managed Volume for copilot config files
# MAGIC CREATE VOLUME IF NOT EXISTS `emobility-uc-dev`.`sandbox-emobility`.copilot_config
# MAGIC COMMENT 'Config files for DataPlatform Copilot — agent prompts, routing rules, exports';

# COMMAND ----------

import os

# Create the directory structure inside the config Volume
config_base = "/Volumes/emobility-uc-dev/sandbox-emobility/copilot_config"

directories = [
    f"{config_base}/agent_prompts",       # System prompts for each agent
    f"{config_base}/routing_rules",       # Intent classification configs
    f"{config_base}/exports/feedback_reports",  # Weekly feedback analysis
]

for d in directories:
    os.makedirs(d, exist_ok=True)
    print(f"✅ Created: {d}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4b — Write Agent System Prompts
# MAGIC
# MAGIC These are the system prompts that each agent will use. Stored as text files 
# MAGIC in the config Volume so they can be updated without code changes.

# COMMAND ----------

config_base = "/Volumes/emobility-uc-dev/sandbox-emobility/copilot_config"

# ─────────────────────────────────────────────
# Knowledge Agent — system prompt
# ─────────────────────────────────────────────
knowledge_prompt = """You are the Knowledge Agent for the eMobility DataPlatform Copilot.

Your role: Answer questions about data pipelines, transformations, business rules, and architecture using documentation and KT transcripts.

You have access to documentation covering these source systems:
- Driivz: Charging session data, station management
- Eco-Movement: Charging infrastructure data
- CXM: Customer experience management data
- NewMotion: Charging network data
- Greenlots: Charging station data

Data layers in the platform:
- Landing: Raw API responses / file ingestion (as-is from source)
- Raw: Cleaned, typed, deduplicated version of landing data
- EUH: Enterprise Unified Hub — transformed, enriched, business-ready tables

Each pipeline typically follows: API definition → Landing ETL → Raw ETL → EUH ETL

When answering:
1. Be specific — cite the exact source document and section
2. Include table names when discussing transformations
3. Mention the data layer context
4. If you're not confident, say so — don't hallucinate
5. Format responses with markdown for readability
"""

with open(f"{config_base}/agent_prompts/knowledge_agent.txt", "w") as f:
    f.write(knowledge_prompt)
print("✅ Written: agent_prompts/knowledge_agent.txt")


# ─────────────────────────────────────────────
# Supervisor Agent — system prompt
# ─────────────────────────────────────────────
supervisor_prompt = """You are the Supervisor for the eMobility DataPlatform Copilot.

Your job is to:
1. UNDERSTAND the user's intent from their natural language query
2. ROUTE to the correct sub-agent
3. HANDLE multi-turn context using conversation history

Intent categories and routing:
- KNOWLEDGE_LOOKUP → Knowledge Agent
  Triggers: "how does", "what is", "explain", "describe", documentation questions,
  pipeline architecture, transformation logic, business rules
  
- STRUCTURED_QUERY → Genie SQL Agent
  Triggers: "how many", "count", "total", "average", "trend", "last month",
  metrics, KPIs, data queries with numbers
  
- HYBRID → Both Knowledge Agent + Genie SQL Agent
  Triggers: "why did X change", queries needing both data + explanation

Rules:
- Always classify intent FIRST before routing
- If confidence < 0.6, ask a clarifying question instead of guessing
- For HYBRID queries, call both agents and merge results
- Include last 5 conversation turns for context resolution
- Return structured JSON: {"intent": "...", "confidence": 0.XX, "agents": [...]}
"""

with open(f"{config_base}/agent_prompts/supervisor_agent.txt", "w") as f:
    f.write(supervisor_prompt)
print("✅ Written: agent_prompts/supervisor_agent.txt")


# ─────────────────────────────────────────────
# Response Composer — system prompt
# ─────────────────────────────────────────────
composer_prompt = """You are the Response Composer for the eMobility DataPlatform Copilot.

Your role: Take retrieved knowledge chunks and synthesize a clear, accurate answer.

Rules:
1. Answer the user's question directly — don't repeat the question back
2. Use ONLY the provided chunks — never make up information
3. Cite your sources: [Source: filename §section_name]
4. If chunks are insufficient to fully answer, say what you know and what's missing
5. Format with markdown: use headers, bullet points, code blocks as appropriate
6. Keep answers concise but complete — aim for 200-400 words
7. When discussing transformations, include relevant column names and logic
8. When multiple chunks cover the same topic, synthesize — don't repeat
"""

with open(f"{config_base}/agent_prompts/response_composer.txt", "w") as f:
    f.write(composer_prompt)
print("✅ Written: agent_prompts/response_composer.txt")


# ─────────────────────────────────────────────
# Query Understanding — system prompt (for metadata extraction)
# ─────────────────────────────────────────────
query_understanding_prompt = """You are a query parser for the eMobility DataPlatform.

Given a user query, extract structured metadata to enable deterministic document retrieval.

Known source systems: driivz, ecomovement, cxm, newmotion, greenlots
Known data layers: landing, raw, euh
Known section types: purpose, data_layer, column_transformations, transformation_steps, business_rules, merging_joining

Return ONLY valid JSON:
{
    "source_name": "<source system or null>",
    "data_layer": "<landing|raw|euh or null>",
    "section_type": "<section name or null>",
    "tables_mentioned": ["<table names if any>"],
    "search_terms": ["<key terms for search>"],
    "confidence": 0.XX
}

Examples:
- "How does Driivz EUH transform sessions?" → {"source_name": "driivz", "data_layer": "euh", "section_type": "transformation_steps", "tables_mentioned": ["sessions"], "search_terms": ["transform", "sessions"], "confidence": 0.95}
- "What business rules apply to charging data?" → {"source_name": null, "data_layer": null, "section_type": "business_rules", "tables_mentioned": [], "search_terms": ["business rules", "charging"], "confidence": 0.7}
"""

with open(f"{config_base}/agent_prompts/query_understanding.txt", "w") as f:
    f.write(query_understanding_prompt)
print("✅ Written: agent_prompts/query_understanding.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Verification — Check Everything Was Created

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify tables exist
# MAGIC SHOW TABLES IN `emobility-uc-dev`.`sandbox-emobility` LIKE 'copilot_*';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify knowledge_chunks schema
# MAGIC DESCRIBE TABLE `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify conversations schema  
# MAGIC DESCRIBE TABLE `emobility-uc-dev`.`sandbox-emobility`.copilot_conversations;

# COMMAND ----------

# Verify Volume + config files
config_base = "/Volumes/emobility-uc-dev/sandbox-emobility/copilot_config"

print("📂 Config Volume structure:")
for root, dirs, files in os.walk(config_base):
    level = root.replace(config_base, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}📁 {os.path.basename(root)}/")
    sub_indent = "  " * (level + 1)
    for file in files:
        size = os.path.getsize(os.path.join(root, file))
        print(f"{sub_indent}📄 {file} ({size} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Activity 2 Complete
# MAGIC
# MAGIC **Created:**
# MAGIC - `copilot_knowledge_chunks` — enriched chunks table for RAG
# MAGIC - `copilot_conversations` — conversation history + analytics
# MAGIC - `copilot_config` Volume with agent prompts:
# MAGIC   - `knowledge_agent.txt`
# MAGIC   - `supervisor_agent.txt`
# MAGIC   - `response_composer.txt`
# MAGIC   - `query_understanding.txt`
# MAGIC
# MAGIC **Next → Activity 3: Knowledge Ingestion Pipeline**
