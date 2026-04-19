# Databricks notebook source
# MAGIC %md
# MAGIC # 📚 Activity 3: Knowledge Ingestion Pipeline
# MAGIC
# MAGIC **Goal**: Read all docs + KT transcripts → chunk by section → enrich with metadata → embed → store in `copilot_knowledge_chunks`.
# MAGIC
# MAGIC **Doc format per source system**:
# MAGIC ```
# MAGIC 1. Source Overview
# MAGIC 2. Notebook: <api_definition>.py
# MAGIC    a. path
# MAGIC    b. notebook purpose
# MAGIC    c. data layer
# MAGIC    d. column level transformation logic
# MAGIC    e. transformation steps
# MAGIC    f. business rules and hardcoded values
# MAGIC    g. deduplication logic
# MAGIC    h. join logic
# MAGIC    i. error handling and data quality
# MAGIC 3. Notebook: landing_etl_<source>_api  (same a–i sections)
# MAGIC 4. Notebook: raw_etl_<source>_api      (same a–i sections)
# MAGIC 5. Notebook: euh_etl_<source>_api       (same a–i sections)
# MAGIC ```
# MAGIC
# MAGIC **Source systems**: driivz, enovos, spirii, uberall

# COMMAND ----------

# MAGIC %pip install tiktoken
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import re
import hashlib
import uuid
from datetime import datetime
from collections import defaultdict
from typing import Optional

import tiktoken

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
CONFIG = {
    # Source paths
    "docs_volume_path": "/Volumes/emobility-uc-dev/landing/docs",
    "kt_volume_path": "/Volumes/emobility-uc-dev/landing/kt_transcripts",  # UPDATE if different

    # Target table
    "target_table": "`emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks",

    # Chunking
    "max_chunk_tokens": 1500,
    "overlap_tokens": 200,

    # Embedding
    "embedding_endpoint": "databricks-bge-large-en",  # Foundation Model endpoint
    "embedding_batch_size": 50,

    # Known source systems (subdirectory names)
    "known_sources": ["driivz", "enovos", "spirii", "uberall"],

    # Known section headers (normalized) — matches the doc format a–i
    "known_sections": [
        "path",
        "notebook_purpose",
        "data_layer",
        "column_transformations",
        "transformation_steps",
        "business_rules",
        "deduplication_logic",
        "join_logic",
        "error_handling",
        "source_overview",
    ],
}

_enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

print("✅ Configuration loaded")
print(f"   Docs path:     {CONFIG['docs_volume_path']}")
print(f"   KT path:       {CONFIG['kt_volume_path']}")
print(f"   Target table:  {CONFIG['target_table']}")
print(f"   Sources:       {CONFIG['known_sources']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📖 Step 3.1 — Discover Source Files

# COMMAND ----------

def discover_files(base_path: str, extensions: list[str]) -> list[dict]:
    """Walk a Volume directory and return metadata for every matching file."""
    files = []
    if not os.path.exists(base_path):
        print(f"⚠️  Path does not exist: {base_path}")
        return files

    for root, dirs, filenames in os.walk(base_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, base_path)
                files.append({
                    "file_name": fname,
                    "full_path": full_path,
                    "rel_path": rel_path,
                    "extension": ext,
                    "size_bytes": os.path.getsize(full_path),
                })
    return files


# Discover docs
doc_files = discover_files(CONFIG["docs_volume_path"], [".md", ".markdown"])
print(f"📄 Docs found: {len(doc_files)}")
for f in doc_files:
    print(f"   {f['rel_path']}  ({f['size_bytes']} bytes)")

# Discover KT transcripts
kt_files = discover_files(CONFIG["kt_volume_path"], [".txt", ".md", ".srt", ".vtt", ".json"])
print(f"\n🎤 KT transcripts found: {len(kt_files)}")
for f in kt_files[:10]:
    print(f"   {f['rel_path']}  ({f['size_bytes']} bytes)")
if not kt_files:
    print(f"   ⚠️  None found at {CONFIG['kt_volume_path']} — update path if needed")

print(f"\n📊 Total files to ingest: {len(doc_files) + len(kt_files)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✂️ Step 3.2 — Section-Based Chunking
# MAGIC
# MAGIC Each doc has a **Source Overview** followed by multiple **Notebook** blocks.
# MAGIC Each notebook block has sections a–i.
# MAGIC We chunk at the section level — each section becomes one chunk.

# COMMAND ----------

def chunk_by_sections(content: str, max_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    Split markdown content into chunks at section boundaries (## or ###).
    Each section becomes one chunk. Oversized sections are split at paragraph
    boundaries with overlap.
    """
    # Split on markdown headers (## or ### or #)
    section_pattern = r'(?=^#{1,4}\s+.+$)'
    raw_sections = re.split(section_pattern, content, flags=re.MULTILINE)
    raw_sections = [s.strip() for s in raw_sections if s.strip()]

    if len(raw_sections) <= 1:
        raw_sections = [content.strip()]

    chunks = []
    chunk_idx = 0

    for section in raw_sections:
        # Extract header text
        header_match = re.match(r'^(#{1,4})\s+(.+)$', section, re.MULTILINE)
        section_header = header_match.group(2).strip() if header_match else "introduction"

        tokens = count_tokens(section)

        if tokens <= max_tokens:
            chunks.append({
                "text": section,
                "section_header": section_header,
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1
        else:
            # Split oversized section at paragraph boundaries with overlap
            sub_chunks = _split_large_section(section, section_header, max_tokens, overlap_tokens, chunk_idx)
            chunks.extend(sub_chunks)
            chunk_idx += len(sub_chunks)

    return chunks


def _split_large_section(text: str, header: str, max_tokens: int, overlap_tokens: int, start_idx: int) -> list[dict]:
    """Split an oversized section into smaller chunks with paragraph-level overlap."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "section_header": header,
                "chunk_index": start_idx + len(chunks),
            })

            # Overlap: keep last paragraph(s) up to overlap_tokens
            overlap_paras = []
            overlap_tok = 0
            for p in reversed(current_chunk):
                p_tok = count_tokens(p)
                if overlap_tok + p_tok > overlap_tokens:
                    break
                overlap_paras.insert(0, p)
                overlap_tok += p_tok

            current_chunk = overlap_paras + [para]
            current_tokens = overlap_tok + para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "section_header": header,
            "chunk_index": start_idx + len(chunks),
        })

    return chunks


print("✅ Chunking functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏷️ Step 3.3 — Metadata Extraction
# MAGIC
# MAGIC Extract structured metadata from file paths and content.
# MAGIC This powers structured-first retrieval in Activity 5.

# COMMAND ----------

def extract_source_name(rel_path: str) -> Optional[str]:
    """
    Extract source system from subdirectory name.
    e.g., 'uberall/landing_etl_uberall_api.md' → 'uberall'
    """
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        candidate = parts[0].lower().strip()
        # Direct match
        if candidate in CONFIG["known_sources"]:
            return candidate
        # Fuzzy match
        for src in CONFIG["known_sources"]:
            if src in candidate or candidate in src:
                return src
    # Fallback: check filename
    fname = parts[-1].lower()
    for src in CONFIG["known_sources"]:
        if src in fname:
            return src
    return None


def extract_data_layer(file_name: str, content: str) -> Optional[str]:
    """
    Extract data layer from filename pattern.
    landing_etl_* → landing,  raw_etl_* → raw,  euh_etl_* → euh
    API definition notebooks (e.g., uberall.py) → None
    """
    fname = file_name.lower()

    if "landing" in fname:
        return "landing"
    elif "raw" in fname:
        return "raw"
    elif "euh" in fname:
        return "euh"

    # Fallback: check content
    content_lower = content[:2000].lower()
    for layer in ["landing", "raw", "euh"]:
        if f"data layer: {layer}" in content_lower or f"**{layer}**" in content_lower:
            return layer

    return None


def extract_notebook_name(file_name: str) -> Optional[str]:
    """Extract notebook name from filename (minus .md extension)."""
    return os.path.splitext(file_name)[0] or None


def classify_document_type(file_name: str, content: str) -> str:
    """
    Determine document type:
    - source_overview: business/source overview doc
    - notebook_doc: per-notebook technical doc (ETL, API definition)
    - kt_transcript: KT transcript
    """
    fname = file_name.lower()

    if any(kw in fname for kw in ["_etl_", "_api", "landing_", "raw_", "euh_"]):
        return "notebook_doc"

    content_lower = content[:1000].lower()
    if any(kw in content_lower for kw in ["source overview", "business overview", "overview of", "introduction"]):
        return "source_overview"

    # If filename matches a source system name directly (e.g., uberall.md, driivz.md)
    name_no_ext = os.path.splitext(fname)[0]
    if name_no_ext in CONFIG["known_sources"]:
        return "source_overview"

    return "notebook_doc"


def extract_tables_mentioned(text: str) -> list[str]:
    """Find table name references: catalog.schema.table, schema.table, and common prefixed tables."""
    tables = set()

    # 3-part: catalog.schema.table
    tables.update(re.findall(r'`?([\w-]+\.[\w-]+\.[\w-]+)`?', text))

    # 2-part: schema.table (filter false positives)
    two_part = re.findall(r'`?([\w-]+\.[\w-]+)`?', text)
    false_positives = {"e.g", "i.e", "v2.0", "v1.0", "0.0", "1.0", "2.0"}
    tables.update(t for t in two_part if t.lower() not in false_positives and not re.match(r'^\d+\.\d+$', t))

    # Prefixed table names in backticks
    tables.update(re.findall(r'`((?:dim_|fact_|stg_|raw_|euh_|landing_)[\w]+)`', text))

    return sorted(list(tables))


def extract_keywords(text: str, top_k: int = 10) -> list[str]:
    """Extract top-k key terms by frequency (stopwords removed)."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "each", "every", "all", "any", "few", "more", "most", "some",
        "this", "that", "these", "those", "what", "which", "who",
        "it", "its", "he", "she", "they", "them", "we", "you",
        "of", "at", "by", "for", "with", "about", "between", "through",
        "during", "before", "after", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "into", "then", "once",
        "here", "there", "when", "where", "why", "how", "than",
        "very", "just", "also", "using", "used", "based", "following",
        "data", "table", "column", "value", "type", "string", "int",
        "null", "true", "false", "none",
    }

    words = re.findall(r'\b[a-zA-Z_]{3,}\b', text.lower())
    freq = defaultdict(int)
    for w in words:
        if w not in stopwords:
            freq[w] += 1

    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:top_k]]


def normalize_section_header(header: str) -> str:
    """
    Normalize section headers to standard names matching the doc format (a–i).
    """
    h = header.lower().strip()
    h = re.sub(r'^#+\s*', '', h)          # Remove leading #
    h = re.sub(r'^[a-i][\.\)]\s*', '', h)  # Remove a., b), etc.
    h = re.sub(r'^\d+[\.\)]\s*', '', h)    # Remove 1., 2), etc.
    h = h.strip()

    # Map variations to standard names
    mappings = {
        "path": "path",
        "notebook path": "path",
        "notebook purpose": "notebook_purpose",
        "purpose": "notebook_purpose",
        "objective": "notebook_purpose",
        "data layer": "data_layer",
        "data_layer": "data_layer",
        "column level transformation logic": "column_transformations",
        "column-level transformation logic": "column_transformations",
        "column level transformations": "column_transformations",
        "column transformations": "column_transformations",
        "transformation steps": "transformation_steps",
        "transformations": "transformation_steps",
        "transformation logic": "transformation_steps",
        "business rules and hardcoded values": "business_rules",
        "business rules": "business_rules",
        "hardcoded values": "business_rules",
        "business logic": "business_rules",
        "deduplication logic": "deduplication_logic",
        "deduplication": "deduplication_logic",
        "deduplicate": "deduplication_logic",
        "join logic": "join_logic",
        "merging/joining logic": "join_logic",
        "merging joining logic": "join_logic",
        "joins": "join_logic",
        "error handling and data quality": "error_handling",
        "error handling": "error_handling",
        "data quality": "error_handling",
        "error handling & data quality": "error_handling",
        "source overview": "source_overview",
        "overview": "source_overview",
        "business overview": "source_overview",
        "introduction": "source_overview",
    }

    for pattern, normalized in mappings.items():
        if pattern in h:
            return normalized

    return h


# Test
print("✅ Metadata extraction functions defined")
print()
print("Tests:")
print(f"  extract_source_name('uberall/euh_etl_uberall.md')          → {extract_source_name('uberall/euh_etl_uberall.md')}")
print(f"  extract_data_layer('landing_etl_spirii_api.md', '')         → {extract_data_layer('landing_etl_spirii_api.md', '')}")
print(f"  extract_notebook_name('raw_etl_enovos_api.md')              → {extract_notebook_name('raw_etl_enovos_api.md')}")
print(f"  classify_document_type('uberall.md', 'Source Overview...')   → {classify_document_type('uberall.md', 'Source Overview of Uberall')}")
print(f"  classify_document_type('euh_etl_driivz.md', '')             → {classify_document_type('euh_etl_driivz.md', '')}")
print(f"  normalize_section_header('## Column Level Transformation Logic') → {normalize_section_header('## Column Level Transformation Logic')}")
print(f"  normalize_section_header('### Business Rules and Hardcoded Values') → {normalize_section_header('### Business Rules and Hardcoded Values')}")
print(f"  normalize_section_header('### Deduplication Logic')         → {normalize_section_header('### Deduplication Logic')}")
print(f"  normalize_section_header('### Error Handling and Data Quality') → {normalize_section_header('### Error Handling and Data Quality')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Step 3.4 + 3.5 + 3.6 — Process, Deduplicate, Embed, Store

# COMMAND ----------

import mlflow.deployments


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings using Databricks Foundation Model endpoint."""
    client = mlflow.deployments.get_deploy_client("databricks")
    all_embeddings = []
    batch_size = CONFIG["embedding_batch_size"]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.predict(
            endpoint=CONFIG["embedding_endpoint"],
            inputs={"input": batch}
        )
        batch_embeddings = [item["embedding"] for item in response["data"]]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def compute_content_hash(text: str) -> str:
    """SHA-256 hash of content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_single_file(file_info: dict, source_type: str) -> list[dict]:
    """
    Full processing pipeline for a single file:
    Read → Chunk → Enrich metadata → Return chunk records.
    """
    try:
        with open(file_info["full_path"], "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"   ❌ Error reading {file_info['rel_path']}: {e}")
        return []

    if not content.strip():
        print(f"   ⚠️  Empty file: {file_info['rel_path']}")
        return []

    # Chunk by sections
    chunks = chunk_by_sections(content, CONFIG["max_chunk_tokens"], CONFIG["overlap_tokens"])

    # File-level metadata
    source_name = extract_source_name(file_info["rel_path"])
    data_layer = extract_data_layer(file_info["file_name"], content)
    notebook_name = extract_notebook_name(file_info["file_name"])
    document_type = classify_document_type(file_info["file_name"], content)

    # If source_type is kt_transcript, override document_type
    if source_type == "kt_transcript":
        document_type = "kt_transcript"

    # Build chunk records
    records = []
    total_chunks = len(chunks)

    for chunk in chunks:
        record = {
            "chunk_id": str(uuid.uuid4()),
            "content": chunk["text"],
            "content_hash": compute_content_hash(chunk["text"]),
            "embedding": None,  # Filled in batch later
            "source_type": source_type,
            "source_name": source_name,
            "document_type": document_type,
            "data_layer": data_layer,
            "notebook_name": notebook_name,
            "section_header": normalize_section_header(chunk["section_header"]),
            "tables_mentioned": extract_tables_mentioned(chunk["text"]),
            "keywords": extract_keywords(chunk["text"]),
            "source_file_path": file_info["full_path"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": total_chunks,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        records.append(record)

    return records


print("✅ Processing functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Run Full Ingestion

# COMMAND ----------

# ─────────────────────────────────────────
# Process ALL files
# ─────────────────────────────────────────
all_records = []

# Process docs
print("📄 Processing documentation files...")
print("=" * 60)
for i, file_info in enumerate(doc_files):
    records = process_single_file(file_info, source_type="doc_generator")
    all_records.extend(records)
    src = extract_source_name(file_info["rel_path"]) or "?"
    print(f"   [{i+1}/{len(doc_files)}] {file_info['rel_path']} → {len(records)} chunks  (source: {src})")

# Process KT transcripts
if kt_files:
    print(f"\n🎤 Processing KT transcripts...")
    print("=" * 60)
    for i, file_info in enumerate(kt_files):
        records = process_single_file(file_info, source_type="kt_transcript")
        all_records.extend(records)
        print(f"   [{i+1}/{len(kt_files)}] {file_info['rel_path']} → {len(records)} chunks")

# ─────────────────────────────────────────
# Stats
# ─────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"📊 TOTAL CHUNKS PRODUCED: {len(all_records)}")

by_source = defaultdict(int)
by_layer = defaultdict(int)
by_type = defaultdict(int)
by_section = defaultdict(int)
for r in all_records:
    by_source[r["source_name"] or "unknown"] += 1
    by_layer[r["data_layer"] or "none"] += 1
    by_type[r["document_type"]] += 1
    by_section[r["section_header"]] += 1

print(f"\n   By source system:")
for k, v in sorted(by_source.items(), key=lambda x: -x[1]):
    print(f"     {k}: {v} chunks")

print(f"\n   By data layer:")
for k, v in sorted(by_layer.items(), key=lambda x: -x[1]):
    print(f"     {k}: {v} chunks")

print(f"\n   By document type:")
for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
    print(f"     {k}: {v} chunks")

print(f"\n   By section header (top 15):")
for k, v in sorted(by_section.items(), key=lambda x: -x[1])[:15]:
    print(f"     {k}: {v} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deduplication

# COMMAND ----------

# Check existing hashes in table (for incremental re-ingestion)
try:
    existing_hashes_df = spark.sql(f"SELECT content_hash FROM {CONFIG['target_table']}")
    existing_hashes = set(row.content_hash for row in existing_hashes_df.collect())
    print(f"   Existing chunks in table: {len(existing_hashes)}")
except Exception:
    existing_hashes = set()
    print(f"   Table is empty — first ingestion")

# Deduplicate
seen_hashes = set()
unique_records = []
dup_in_batch = 0
dup_in_table = 0

for r in all_records:
    h = r["content_hash"]
    if h in seen_hashes:
        dup_in_batch += 1
    elif h in existing_hashes:
        dup_in_table += 1
    else:
        seen_hashes.add(h)
        unique_records.append(r)

print(f"   Duplicates within batch:   {dup_in_batch}")
print(f"   Already in table (skipped): {dup_in_table}")
print(f"   ✅ New unique chunks:       {len(unique_records)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Embeddings

# COMMAND ----------

if unique_records:
    print(f"🧠 Generating embeddings for {len(unique_records)} chunks...")
    print(f"   Endpoint: {CONFIG['embedding_endpoint']}")
    print(f"   Batch size: {CONFIG['embedding_batch_size']}")

    texts = [r["content"] for r in unique_records]

    try:
        embeddings = get_embeddings(texts)
        for record, emb in zip(unique_records, embeddings):
            record["embedding"] = emb

        print(f"   ✅ Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")
    except Exception as e:
        print(f"   ❌ Embedding error: {e}")
        print(f"   Continuing without embeddings — re-run this cell after fixing the endpoint.")
else:
    print("⚠️  No new chunks to embed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Delta Table

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType,
    IntegerType, TimestampType
)

if unique_records:
    print(f"💾 Writing {len(unique_records)} chunks to {CONFIG['target_table']}...")

    schema = StructType([
        StructField("chunk_id", StringType(), False),
        StructField("content", StringType(), False),
        StructField("content_hash", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), True),
        StructField("source_type", StringType(), True),
        StructField("source_name", StringType(), True),
        StructField("document_type", StringType(), True),
        StructField("data_layer", StringType(), True),
        StructField("notebook_name", StringType(), True),
        StructField("section_header", StringType(), True),
        StructField("tables_mentioned", ArrayType(StringType()), True),
        StructField("keywords", ArrayType(StringType()), True),
        StructField("source_file_path", StringType(), True),
        StructField("chunk_index", IntegerType(), True),
        StructField("total_chunks", IntegerType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
    ])

    df = spark.createDataFrame(unique_records, schema=schema)
    df.write.mode("append").saveAsTable(CONFIG["target_table"])

    print(f"   ✅ Written {df.count()} chunks")
else:
    print("⚠️  No records to write")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Step 3.7 — Verification

# COMMAND ----------

total = spark.sql(f"SELECT COUNT(*) as cnt FROM {CONFIG['target_table']}").first().cnt
print(f"📊 Total chunks in table: {total}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Chunks by source system
# MAGIC SELECT source_name, COUNT(*) as chunks, COUNT(DISTINCT notebook_name) as notebooks
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks
# MAGIC GROUP BY source_name ORDER BY chunks DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Chunks by data layer
# MAGIC SELECT data_layer, COUNT(*) as chunks
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks
# MAGIC GROUP BY data_layer ORDER BY chunks DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Chunks by section header (should see all 9 section types + source_overview)
# MAGIC SELECT section_header, COUNT(*) as chunks
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks
# MAGIC GROUP BY section_header ORDER BY chunks DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Spot-check: 5 random chunks with metadata
# MAGIC SELECT
# MAGIC     chunk_id, source_name, data_layer, notebook_name, section_header,
# MAGIC     tables_mentioned, keywords,
# MAGIC     LEFT(content, 200) as content_preview,
# MAGIC     CASE WHEN embedding IS NOT NULL THEN 'yes' ELSE 'no' END as has_embedding
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks
# MAGIC ORDER BY RAND() LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Embedding coverage
# MAGIC SELECT
# MAGIC     COUNT(*) as total,
# MAGIC     SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding,
# MAGIC     SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) as without_embedding
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks;

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Activity 3 Complete
# MAGIC
# MAGIC **What was done:**
# MAGIC - Read `.md` docs for: driivz, enovos, spirii, uberall
# MAGIC - Chunked by section headers (path, purpose, data layer, column transformations,
# MAGIC   transformation steps, business rules, deduplication logic, join logic, error handling)
# MAGIC - Enriched each chunk with structured metadata
# MAGIC - Deduplicated by content hash (safe to re-run)
# MAGIC - Generated BGE-large-en embeddings
# MAGIC - Stored in `copilot_knowledge_chunks`
# MAGIC
# MAGIC **Check the verification output above:**
# MAGIC 1. Are chunk counts per source reasonable?
# MAGIC 2. Are all 4 sources present? (driivz, enovos, spirii, uberall)
# MAGIC 3. Are section headers correct? (should see ~9 types + source_overview)
# MAGIC 4. Do embeddings exist for all chunks?
# MAGIC
# MAGIC **Next → Activity 4: Create Vector Search Index**
