# Databricks notebook source
# MAGIC %md
# MAGIC # 📚 Activity 3: Knowledge Ingestion Pipeline
# MAGIC
# MAGIC **Goal**: Read all docs + KT transcripts → chunk by section → enrich with metadata → embed → store in `copilot_knowledge_chunks`.
# MAGIC
# MAGIC **Pipeline flow**:
# MAGIC ```
# MAGIC Source files (.md) → Section-based chunking → Metadata enrichment → Deduplication → Embedding → Delta table
# MAGIC ```
# MAGIC
# MAGIC **Tasks**:
# MAGIC - 3.1 — Read docs from Volumes
# MAGIC - 3.2 — Semantic chunking (by section headers)
# MAGIC - 3.3 — Metadata extraction (source_name, data_layer, notebook_name, section_header, tables_mentioned, keywords)
# MAGIC - 3.4 — Deduplication (content_hash)
# MAGIC - 3.5 — Embed chunks using Foundation Model
# MAGIC - 3.6 — Write to `copilot_knowledge_chunks`
# MAGIC - 3.7 — Verify quality

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration

# COMMAND ----------

# MAGIC %pip install tiktoken
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import re
import json
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
    "embedding_dimension": 1024,
    "embedding_batch_size": 50,

    # Known source systems (from subdirectory names)
    "known_sources": ["driivz", "ecomovement", "cxm", "newmotion", "greenlots"],

    # Known section headers in docs (normalized lowercase)
    "known_sections": [
        "purpose",
        "data layer",
        "column level transformation logic",
        "column_transformations",
        "transformation steps",
        "business rules",
        "merging/joining logic",
        "merging_joining",
        "overview",
        "source schema",
        "target schema",
    ],
}

# Token counter
_enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

print("✅ Configuration loaded")
print(f"   Docs path:   {CONFIG['docs_volume_path']}")
print(f"   KT path:     {CONFIG['kt_volume_path']}")
print(f"   Target:      {CONFIG['target_table']}")
print(f"   Max chunk:   {CONFIG['max_chunk_tokens']} tokens")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📖 Step 3.1 — Read All Source Files

# COMMAND ----------

def discover_files(base_path: str, extensions: list[str]) -> list[dict]:
    """
    Walk a Volume directory and return metadata for every matching file.
    """
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
for f in doc_files[:20]:
    print(f"   {f['rel_path']}  ({f['size_bytes']} bytes)")
if len(doc_files) > 20:
    print(f"   ... and {len(doc_files) - 20} more")

# Discover KT transcripts
kt_files = discover_files(CONFIG["kt_volume_path"], [".txt", ".md", ".srt", ".vtt", ".json"])
print(f"\n🎤 KT transcripts found: {len(kt_files)}")
for f in kt_files[:10]:
    print(f"   {f['rel_path']}  ({f['size_bytes']} bytes)")
if len(kt_files) > 10:
    print(f"   ... and {len(kt_files) - 10} more")

if not kt_files:
    print(f"   ⚠️  No KT text files found at {CONFIG['kt_volume_path']}")
    print(f"   If KT transcripts are elsewhere, update CONFIG['kt_volume_path'] and re-run.")

print(f"\n📊 Total files to ingest: {len(doc_files) + len(kt_files)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✂️ Step 3.2 — Section-Based Chunking
# MAGIC
# MAGIC Chunks docs at section boundaries (respecting markdown headers).
# MAGIC Your docs consistently have these sections per notebook:
# MAGIC - Purpose
# MAGIC - Data Layer
# MAGIC - Column Level Transformation Logic
# MAGIC - Transformation Steps
# MAGIC - Business Rules
# MAGIC - Merging/Joining Logic
# MAGIC
# MAGIC Each section becomes **one chunk**. If a section exceeds the token limit, it gets split with overlap.

# COMMAND ----------

def chunk_by_sections(content: str, max_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    Split markdown content into chunks at section boundaries.

    Returns list of dicts with keys: text, section_header, chunk_index.
    """
    # Split on markdown headers (## or ###)
    # Keep the header with its section content
    section_pattern = r'(?=^#{1,4}\s+.+$)'
    raw_sections = re.split(section_pattern, content, flags=re.MULTILINE)

    # Clean up: remove empty sections
    raw_sections = [s.strip() for s in raw_sections if s.strip()]

    # If no headers found, treat entire content as one section
    if len(raw_sections) <= 1:
        raw_sections = [content.strip()]

    chunks = []
    chunk_idx = 0

    for section in raw_sections:
        # Extract section header
        header_match = re.match(r'^(#{1,4})\s+(.+)$', section, re.MULTILINE)
        if header_match:
            section_header = header_match.group(2).strip()
        else:
            section_header = "introduction"

        tokens = count_tokens(section)

        if tokens <= max_tokens:
            # Section fits in one chunk
            chunks.append({
                "text": section,
                "section_header": section_header,
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1
        else:
            # Section too large — split by paragraphs with overlap
            sub_chunks = _split_large_section(section, section_header, max_tokens, overlap_tokens, chunk_idx)
            chunks.extend(sub_chunks)
            chunk_idx += len(sub_chunks)

    return chunks


def _split_large_section(text: str, header: str, max_tokens: int, overlap_tokens: int, start_idx: int) -> list[dict]:
    """
    Split an oversized section into smaller chunks, keeping paragraph boundaries where possible.
    Adds overlap between consecutive chunks for context continuity.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "section_header": header,
                "chunk_index": start_idx + len(chunks),
            })

            # Start new chunk with overlap — keep last paragraph(s)
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

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "section_header": header,
            "chunk_index": start_idx + len(chunks),
        })

    return chunks


# Quick test
test_md = """# Overview
This is the overview section.

## Purpose
This notebook handles landing ETL for Driivz API.

## Transformation Steps
Step 1: Call the API
Step 2: Parse response
Step 3: Write to landing table
"""

test_chunks = chunk_by_sections(test_md, max_tokens=1500, overlap_tokens=200)
print(f"✅ Chunking function works — test doc produced {len(test_chunks)} chunks:")
for c in test_chunks:
    print(f"   [{c['chunk_index']}] §{c['section_header']} ({count_tokens(c['text'])} tokens)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏷️ Step 3.3 — Metadata Extraction
# MAGIC
# MAGIC Extract structured metadata from file paths and content for each chunk.
# MAGIC This powers the structured-first retrieval in Activity 5.

# COMMAND ----------

def extract_source_name(rel_path: str) -> Optional[str]:
    """
    Extract source system name from the subdirectory.
    e.g., 'driivz/euh_etl_driivz.md' → 'driivz'
    """
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        candidate = parts[0].lower().strip()
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
    landing_etl_* → landing
    raw_etl_*     → raw
    euh_etl_*     → euh
    """
    fname = file_name.lower()

    if "landing" in fname or "landing_etl" in fname:
        return "landing"
    elif "raw" in fname or "raw_etl" in fname:
        return "raw"
    elif "euh" in fname or "euh_etl" in fname:
        return "euh"

    # Fallback: check content for explicit layer mentions
    content_lower = content[:2000].lower()  # Check just the beginning
    if "data layer: landing" in content_lower or "**landing**" in content_lower:
        return "landing"
    elif "data layer: raw" in content_lower or "**raw**" in content_lower:
        return "raw"
    elif "data layer: euh" in content_lower or "**euh**" in content_lower:
        return "euh"

    return None


def extract_notebook_name(file_name: str) -> Optional[str]:
    """
    Extract notebook name from filename.
    e.g., 'landing_etl_driivz_api.md' → 'landing_etl_driivz_api'
    """
    name = os.path.splitext(file_name)[0]
    return name if name else None


def classify_document_type(file_name: str, content: str) -> str:
    """
    Determine if this is a business overview or a notebook-level doc.
    Business overviews typically don't have ETL-related names.
    """
    fname = file_name.lower()

    # Notebook docs have etl/api patterns
    if any(kw in fname for kw in ["_etl_", "_api", "landing_", "raw_", "euh_"]):
        return "notebook_doc"

    # Check content for business overview indicators
    content_lower = content[:1000].lower()
    if any(kw in content_lower for kw in ["business overview", "company overview", "about ", "introduction to"]):
        return "business_overview"

    # Default to notebook_doc
    return "notebook_doc"


def extract_tables_mentioned(text: str) -> list[str]:
    """
    Find table name references in the text.
    Looks for catalog.schema.table patterns and common table name patterns.
    """
    tables = set()

    # Pattern 1: catalog.schema.table (3-part names)
    three_part = re.findall(r'`?([\w-]+\.[\w-]+\.[\w-]+)`?', text)
    tables.update(three_part)

    # Pattern 2: schema.table (2-part names)
    two_part = re.findall(r'`?([\w-]+\.[\w-]+)`?', text)
    # Filter out common false positives
    false_positives = {"e.g", "i.e", "v2.0", "v1.0", "0.0", "1.0", "2.0"}
    two_part = [t for t in two_part if t.lower() not in false_positives and not re.match(r'^\d+\.\d+$', t)]
    tables.update(two_part)

    # Pattern 3: explicit table mentions (e.g., "table: dim_station" or "`fact_sessions`")
    explicit = re.findall(r'`((?:dim_|fact_|stg_|raw_|euh_|landing_)[\w]+)`', text)
    tables.update(explicit)

    return sorted(list(tables))


def extract_keywords(text: str, top_k: int = 10) -> list[str]:
    """
    Extract key terms from text using simple frequency-based approach.
    (Not using sklearn TF-IDF to avoid extra dependency — this works well enough for our structured docs.)
    """
    # Common stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "ought",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "through",
        "during", "before", "after", "above", "below", "to", "from", "up",
        "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "this", "that", "these", "those", "what", "which", "who", "whom",
        "it", "its", "he", "she", "they", "them", "his", "her", "their",
        "we", "you", "me", "him", "us", "my", "your", "our",
        "data", "table", "column", "value", "type", "string", "int", "null",
        "true", "false", "using", "used", "also", "based", "following",
    }

    # Tokenize: extract words 3+ chars
    words = re.findall(r'\b[a-zA-Z_]{3,}\b', text.lower())

    # Count frequency, excluding stopwords
    freq = defaultdict(int)
    for w in words:
        if w not in stopwords:
            freq[w] += 1

    # Return top-k by frequency
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:top_k]]


def normalize_section_header(header: str) -> str:
    """
    Normalize section headers to consistent naming.
    """
    h = header.lower().strip()
    h = re.sub(r'^#+\s*', '', h)      # Remove leading #'s
    h = re.sub(r'^\d+[\.\)]\s*', '', h)  # Remove numbering like "1." or "2)"
    h = h.strip()

    # Map common variations to standard names
    mappings = {
        "purpose": "purpose",
        "objective": "purpose",
        "overview": "overview",
        "business overview": "overview",
        "introduction": "overview",
        "data layer": "data_layer",
        "data_layer": "data_layer",
        "column level transformation logic": "column_transformations",
        "column-level transformation logic": "column_transformations",
        "column transformations": "column_transformations",
        "column level transformations": "column_transformations",
        "transformation steps": "transformation_steps",
        "transformations": "transformation_steps",
        "transformation logic": "transformation_steps",
        "business rules": "business_rules",
        "business logic": "business_rules",
        "merging/joining logic": "merging_joining",
        "merging joining logic": "merging_joining",
        "join logic": "merging_joining",
        "joins": "merging_joining",
        "source schema": "source_schema",
        "target schema": "target_schema",
    }

    for pattern, normalized in mappings.items():
        if pattern in h:
            return normalized

    return h


# Test metadata extraction
print("✅ Metadata extraction functions defined")
print(f"\n   extract_source_name('driivz/euh_etl_driivz.md') → {extract_source_name('driivz/euh_etl_driivz.md')}")
print(f"   extract_data_layer('landing_etl_driivz_api.md', '') → {extract_data_layer('landing_etl_driivz_api.md', '')}")
print(f"   extract_notebook_name('euh_etl_driivz.md') → {extract_notebook_name('euh_etl_driivz.md')}")
print(f"   classify_document_type('euh_etl_driivz.md', '') → {classify_document_type('euh_etl_driivz.md', '')}")
print(f"   normalize_section_header('## Column Level Transformation Logic') → {normalize_section_header('## Column Level Transformation Logic')}")
print(f"   extract_tables_mentioned('reads from `raw.driivz.sessions` and writes to `euh.driivz.fact_sessions`')")
print(f"     → {extract_tables_mentioned('reads from `raw.driivz.sessions` and writes to `euh.driivz.fact_sessions`')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Step 3.4 + 3.5 + 3.6 — Full Processing Pipeline
# MAGIC
# MAGIC For each file:
# MAGIC 1. Read content
# MAGIC 2. Chunk by sections
# MAGIC 3. Enrich each chunk with metadata
# MAGIC 4. Compute content hash (dedup)
# MAGIC 5. Embed using Foundation Model
# MAGIC 6. Write to Delta table

# COMMAND ----------

import mlflow.deployments


def get_embeddings(texts: list[str], endpoint: str = CONFIG["embedding_endpoint"]) -> list[list[float]]:
    """
    Get embeddings for a batch of texts using Databricks Foundation Model endpoint.
    """
    client = mlflow.deployments.get_deploy_client("databricks")

    # Process in batches (API may have input limits)
    all_embeddings = []
    batch_size = CONFIG["embedding_batch_size"]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.predict(
            endpoint=endpoint,
            inputs={"input": batch}
        )

        # Extract embeddings from response
        batch_embeddings = [item["embedding"] for item in response["data"]]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def compute_content_hash(text: str) -> str:
    """SHA-256 hash of content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_single_file(file_info: dict, source_type: str) -> list[dict]:
    """
    Full processing pipeline for a single file.
    Returns a list of chunk records ready for insertion.
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

    # 1. Chunk by sections
    chunks = chunk_by_sections(content, CONFIG["max_chunk_tokens"], CONFIG["overlap_tokens"])

    # 2. Extract file-level metadata
    source_name = extract_source_name(file_info["rel_path"])
    data_layer = extract_data_layer(file_info["file_name"], content)
    notebook_name = extract_notebook_name(file_info["file_name"])
    document_type = classify_document_type(file_info["file_name"], content)

    # 3. Build chunk records
    records = []
    total_chunks = len(chunks)

    for chunk in chunks:
        text = chunk["text"]
        section_raw = chunk["section_header"]

        record = {
            "chunk_id": str(uuid.uuid4()),
            "content": text,
            "content_hash": compute_content_hash(text),
            "embedding": None,  # Filled in batch later
            "source_type": source_type,
            "source_name": source_name,
            "document_type": document_type,
            "data_layer": data_layer,
            "notebook_name": notebook_name,
            "section_header": normalize_section_header(section_raw),
            "tables_mentioned": extract_tables_mentioned(text),
            "keywords": extract_keywords(text),
            "source_file_path": file_info["full_path"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": total_chunks,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        records.append(record)

    return records


# Quick test with first doc
if doc_files:
    test_records = process_single_file(doc_files[0], source_type="doc_generator")
    print(f"✅ Processing test — {doc_files[0]['rel_path']}:")
    print(f"   Produced {len(test_records)} chunks")
    if test_records:
        r = test_records[0]
        print(f"   First chunk preview:")
        print(f"     source_name:    {r['source_name']}")
        print(f"     data_layer:     {r['data_layer']}")
        print(f"     notebook_name:  {r['notebook_name']}")
        print(f"     document_type:  {r['document_type']}")
        print(f"     section_header: {r['section_header']}")
        print(f"     tables:         {r['tables_mentioned'][:5]}")
        print(f"     keywords:       {r['keywords']}")
        print(f"     tokens:         {count_tokens(r['content'])}")
        print(f"     content_hash:   {r['content_hash'][:16]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Step 3.6 — Run Full Ingestion

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType,
    IntegerType, TimestampType
)

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
    print(f"   [{i+1}/{len(doc_files)}] {file_info['rel_path']} → {len(records)} chunks")

# Process KT transcripts
if kt_files:
    print(f"\n🎤 Processing KT transcripts...")
    print("=" * 60)
    for i, file_info in enumerate(kt_files):
        records = process_single_file(file_info, source_type="kt_transcript")
        all_records.extend(records)
        print(f"   [{i+1}/{len(kt_files)}] {file_info['rel_path']} → {len(records)} chunks")

print(f"\n📊 TOTAL CHUNKS PRODUCED: {len(all_records)}")

# Stats
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

print(f"\n   By section (top 15):")
for k, v in sorted(by_section.items(), key=lambda x: -x[1])[:15]:
    print(f"     {k}: {v} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 — Deduplication
# MAGIC
# MAGIC Remove duplicate chunks (same content hash). This handles re-ingestion cleanly.

# COMMAND ----------

# ─────────────────────────────────────────
# Deduplicate by content hash
# ─────────────────────────────────────────
print("🔍 Deduplicating...")

# Check existing hashes in the table (for incremental ingestion)
try:
    existing_hashes_df = spark.sql(f"SELECT content_hash FROM {CONFIG['target_table']}")
    existing_hashes = set(row.content_hash for row in existing_hashes_df.collect())
    print(f"   Existing chunks in table: {len(existing_hashes)}")
except Exception:
    existing_hashes = set()
    print(f"   Table is empty — first ingestion")

# Deduplicate within this batch
seen_hashes = set()
unique_records = []
duplicate_count = 0
skip_existing = 0

for r in all_records:
    h = r["content_hash"]
    if h in seen_hashes:
        duplicate_count += 1
    elif h in existing_hashes:
        skip_existing += 1
    else:
        seen_hashes.add(h)
        unique_records.append(r)

print(f"   Duplicates within batch: {duplicate_count}")
print(f"   Already in table (skipped): {skip_existing}")
print(f"   New unique chunks to ingest: {len(unique_records)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 — Generate Embeddings
# MAGIC
# MAGIC Embed all unique chunks using the BGE-large-en Foundation Model.

# COMMAND ----------

# ─────────────────────────────────────────
# Generate embeddings in batches
# ─────────────────────────────────────────
if unique_records:
    print(f"🧠 Generating embeddings for {len(unique_records)} chunks...")
    print(f"   Model: {CONFIG['embedding_endpoint']}")
    print(f"   Batch size: {CONFIG['embedding_batch_size']}")

    texts = [r["content"] for r in unique_records]

    try:
        embeddings = get_embeddings(texts)
        print(f"   ✅ Generated {len(embeddings)} embeddings")
        print(f"   Dimension: {len(embeddings[0])}")

        # Attach embeddings to records
        for record, emb in zip(unique_records, embeddings):
            record["embedding"] = emb

    except Exception as e:
        print(f"   ❌ Embedding error: {e}")
        print(f"   Setting embeddings to None — you can re-run this cell after fixing the endpoint")
        for record in unique_records:
            record["embedding"] = None
else:
    print("⚠️  No new chunks to embed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 — Write to Delta Table

# COMMAND ----------

# ─────────────────────────────────────────
# Write to copilot_knowledge_chunks
# ─────────────────────────────────────────
if unique_records:
    print(f"💾 Writing {len(unique_records)} chunks to {CONFIG['target_table']}...")

    # Define schema
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

    # Convert to Spark DataFrame
    df = spark.createDataFrame(unique_records, schema=schema)

    # Append to table
    df.write.mode("append").saveAsTable(CONFIG["target_table"])

    print(f"   ✅ Successfully written {df.count()} chunks")
else:
    print("⚠️  No records to write")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Step 3.7 — Verify Quality

# COMMAND ----------

# ─────────────────────────────────────────
# Verification queries
# ─────────────────────────────────────────
print("🔍 VERIFICATION")
print("=" * 60)

# Total count
total = spark.sql(f"SELECT COUNT(*) as cnt FROM {CONFIG['target_table']}").first().cnt
print(f"\n📊 Total chunks in table: {total}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Chunks by source system
# MAGIC SELECT
# MAGIC     source_name,
# MAGIC     COUNT(*) as chunk_count,
# MAGIC     ROUND(AVG(LENGTH(content))) as avg_chars,
# MAGIC     COUNT(DISTINCT notebook_name) as notebooks,
# MAGIC     COUNT(DISTINCT section_header) as sections
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks
# MAGIC GROUP BY source_name
# MAGIC ORDER BY chunk_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Chunks by data layer
# MAGIC SELECT
# MAGIC     data_layer,
# MAGIC     COUNT(*) as chunk_count
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks
# MAGIC GROUP BY data_layer
# MAGIC ORDER BY chunk_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Chunks by section header (are all expected sections present?)
# MAGIC SELECT
# MAGIC     section_header,
# MAGIC     COUNT(*) as chunk_count
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks
# MAGIC GROUP BY section_header
# MAGIC ORDER BY chunk_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Spot-check: view 5 random chunks with full metadata
# MAGIC SELECT
# MAGIC     chunk_id,
# MAGIC     source_name,
# MAGIC     data_layer,
# MAGIC     notebook_name,
# MAGIC     section_header,
# MAGIC     tables_mentioned,
# MAGIC     keywords,
# MAGIC     LEFT(content, 200) as content_preview,
# MAGIC     CASE WHEN embedding IS NOT NULL THEN 'yes' ELSE 'no' END as has_embedding
# MAGIC FROM `emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks
# MAGIC ORDER BY RAND()
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify embeddings were generated
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
# MAGIC - Read all `.md` docs from `/Volumes/emobility-uc-dev/landing/docs/`
# MAGIC - Read KT transcripts (if found)
# MAGIC - Chunked by section headers (purpose, transformations, business rules, etc.)
# MAGIC - Enriched each chunk with: source_name, data_layer, notebook_name, section_header, tables_mentioned, keywords
# MAGIC - Deduplicated by content hash
# MAGIC - Generated embeddings using BGE-large-en
# MAGIC - Stored in `copilot_knowledge_chunks` table
# MAGIC
# MAGIC **Verify the output above:**
# MAGIC 1. Are the chunk counts reasonable?
# MAGIC 2. Are source_name values correct for each subdirectory?
# MAGIC 3. Are data_layer values correct (landing/raw/euh)?
# MAGIC 4. Are section_headers properly normalized?
# MAGIC 5. Do embeddings exist for all chunks?
# MAGIC
# MAGIC **Next → Activity 4: Create Vector Search Index**
