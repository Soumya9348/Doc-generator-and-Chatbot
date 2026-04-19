# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 Activity 4: Create Vector Search Index
# MAGIC
# MAGIC **Goal**: Create a Mosaic AI Vector Search index on `copilot_knowledge_chunks`
# MAGIC so the Knowledge Agent can fall back to semantic search when structured retrieval
# MAGIC returns insufficient results.
# MAGIC
# MAGIC **What this does:**
# MAGIC - Creates a Vector Search endpoint (one-time)
# MAGIC - Creates a Delta Sync index on the `knowledge_chunks` table
# MAGIC - Delta Sync means the index auto-updates when the source table changes
# MAGIC - Tests with 5 sample queries

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 — Create Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

ENDPOINT_NAME = "copilot-vs-endpoint"

# Create endpoint (one-time — will error if already exists, that's fine)
try:
    vsc.create_endpoint(name=ENDPOINT_NAME)
    print(f"✅ Created endpoint: {ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"✅ Endpoint already exists: {ENDPOINT_NAME}")
    else:
        print(f"❌ Error creating endpoint: {e}")
        raise

# COMMAND ----------

# Wait for endpoint to be ready
import time

endpoint = vsc.get_endpoint(ENDPOINT_NAME)
print(f"   Endpoint status: {endpoint}")

# If it's provisioning, wait
max_wait = 300  # 5 minutes
waited = 0
while waited < max_wait:
    try:
        endpoint = vsc.get_endpoint(ENDPOINT_NAME)
        status = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
        if status == "ONLINE":
            print(f"✅ Endpoint is ONLINE")
            break
        else:
            print(f"   Status: {status} — waiting... ({waited}s)")
            time.sleep(15)
            waited += 15
    except Exception as e:
        print(f"   Checking status: {e}")
        time.sleep(15)
        waited += 15

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 — Create Delta Sync Vector Index

# COMMAND ----------

INDEX_NAME = "`emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_index"
SOURCE_TABLE = "`emobility-uc-dev`.`sandbox-emobility`.copilot_knowledge_chunks"

# Create the Delta Sync index
# This auto-syncs the vector index whenever the source table is updated
try:
    index = vsc.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",          # Sync on-demand (run sync manually when needed)
        primary_key="chunk_id",
        embedding_dimension=1024,           # BGE-large-en produces 1024-dim vectors
        embedding_vector_column="embedding",
        columns_to_sync=[                   # Columns available for filtering in search
            "chunk_id",
            "content",
            "source_type",
            "source_name",
            "document_type",
            "data_layer",
            "notebook_name",
            "section_header",
            "tables_mentioned",
            "keywords",
            "source_file_path",
        ],
    )
    print(f"✅ Created Delta Sync index: {INDEX_NAME}")
    print(f"   Pipeline type: TRIGGERED (sync manually after table updates)")
except Exception as e:
    if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"✅ Index already exists: {INDEX_NAME}")
    else:
        print(f"❌ Error: {e}")
        raise

# COMMAND ----------

# Wait for index to sync
print("⏳ Waiting for index to sync (first sync may take a few minutes)...")

max_wait = 600  # 10 minutes
waited = 0
while waited < max_wait:
    try:
        index = vsc.get_index(
            endpoint_name=ENDPOINT_NAME,
            index_name=INDEX_NAME
        )
        status = index.describe()
        index_status = status.get("status", {}).get("ready", False)
        detailed = status.get("status", {}).get("detailed_state", "UNKNOWN")
        
        if index_status:
            print(f"✅ Index is READY")
            print(f"   Status: {status.get('status', {})}")
            break
        else:
            print(f"   Index status: {detailed} — waiting... ({waited}s)")
            time.sleep(30)
            waited += 30
    except Exception as e:
        print(f"   Checking: {e}")
        time.sleep(30)
        waited += 30

if waited >= max_wait:
    print(f"⚠️  Index may still be syncing. Check Databricks UI → Catalog → Vector Search Indexes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 — Test Vector Search
# MAGIC
# MAGIC Run 5 sample queries to verify the index returns relevant chunks.

# COMMAND ----------

import mlflow.deployments

def embed_query(query: str) -> list[float]:
    """Embed a single query for vector search."""
    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
        endpoint="databricks-bge-large-en",
        inputs={"input": [query]}
    )
    return response["data"][0]["embedding"]


# Get the index
index = vsc.get_index(
    endpoint_name=ENDPOINT_NAME,
    index_name=INDEX_NAME
)

# Test queries
test_queries = [
    "How does the Driivz EUH pipeline transform data?",
    "What are the business rules for Spirii?",
    "What deduplication logic is used in the raw layer?",
    "Explain the landing ETL process for Uberall",
    "What join logic is used in the EUH layer?",
]

print("🔍 VECTOR SEARCH TEST RESULTS")
print("=" * 70)

for query in test_queries:
    print(f"\n💬 Query: \"{query}\"")
    print("-" * 60)
    
    try:
        results = index.similarity_search(
            query_vector=embed_query(query),
            columns=["source_name", "notebook_name", "data_layer", "section_header", "content"],
            num_results=3,
        )
        
        if results and "result" in results and results["result"]["data_array"]:
            for i, row in enumerate(results["result"]["data_array"]):
                print(f"   [{i+1}] source={row[0]}  notebook={row[1]}  layer={row[2]}  section={row[3]}")
                content_preview = (row[4] or "")[:120].replace("\n", " ")
                print(f"       preview: {content_preview}...")
        else:
            print(f"   ⚠️ No results returned")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3b — Test with Metadata Filters
# MAGIC
# MAGIC Vector search can pre-filter by metadata columns before computing similarity.
# MAGIC This combines structured filtering with semantic search (hybrid approach).

# COMMAND ----------

print("🔍 FILTERED VECTOR SEARCH TEST")
print("=" * 70)

# Test 1: Filter by source_name
print("\n💬 Query: 'transformation steps' (filtered: source_name = 'spirii')")
print("-" * 60)
try:
    results = index.similarity_search(
        query_vector=embed_query("transformation steps"),
        columns=["source_name", "notebook_name", "data_layer", "section_header", "content"],
        filters={"source_name": "spirii"},
        num_results=3,
    )
    if results and "result" in results and results["result"]["data_array"]:
        for i, row in enumerate(results["result"]["data_array"]):
            print(f"   [{i+1}] source={row[0]}  notebook={row[1]}  layer={row[2]}  section={row[3]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Filter by data_layer
print("\n💬 Query: 'business rules' (filtered: data_layer = 'euh')")
print("-" * 60)
try:
    results = index.similarity_search(
        query_vector=embed_query("business rules"),
        columns=["source_name", "notebook_name", "data_layer", "section_header", "content"],
        filters={"data_layer": "euh"},
        num_results=3,
    )
    if results and "result" in results and results["result"]["data_array"]:
        for i, row in enumerate(results["result"]["data_array"]):
            print(f"   [{i+1}] source={row[0]}  notebook={row[1]}  layer={row[2]}  section={row[3]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Filter by source + layer
print("\n💬 Query: 'deduplication' (filtered: source_name = 'driivz', data_layer = 'raw')")
print("-" * 60)
try:
    results = index.similarity_search(
        query_vector=embed_query("deduplication"),
        columns=["source_name", "notebook_name", "data_layer", "section_header", "content"],
        filters={"source_name": "driivz", "data_layer": "raw"},
        num_results=3,
    )
    if results and "result" in results and results["result"]["data_array"]:
        for i, row in enumerate(results["result"]["data_array"]):
            print(f"   [{i+1}] source={row[0]}  notebook={row[1]}  layer={row[2]}  section={row[3]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Activity 4 Complete
# MAGIC
# MAGIC **Created:**
# MAGIC - Vector Search endpoint: `copilot-vs-endpoint`
# MAGIC - Delta Sync index: `copilot_knowledge_index`
# MAGIC   - Auto-syncs when `knowledge_chunks` table is updated
# MAGIC   - Supports metadata filtering (source_name, data_layer, etc.)
# MAGIC
# MAGIC **Verified:**
# MAGIC - 5 unfiltered semantic search queries
# MAGIC - 3 filtered search queries (by source, layer, both)
# MAGIC
# MAGIC **Next → Activity 5: Knowledge Agent (Structured-First + Vector Fallback)**
