# Architecture & Strategy Pitch: eMobility DataPlatform Copilot

## 1. The Core Problem We Are Solving

We are facing two massive bottlenecks in our data platform operations that are scaling linearly with our growth:

**A. The Tribal Knowledge Crisis (Engineering Bottleneck)**
Our EUH and Raw pipelines for sources like Spirii, Driivz, and Uberall are highly complex. Transformation logic, column derivations, and business rules are buried in individual `.md` files or isolated in different notebooks. When a pipeline fails or a new engineer joins, they spend hours parsing through disjointed documentation or pinging senior engineers to understand *why* a specific deduplication logic exists. Knowledge is siloed and searchability is zero.

**B. The "Data Fetch" Fatigue (Business Bottleneck)**
Our data team is constantly barraged with basic ad-hoc requests: *"How many active stations are in Germany?"*, *"What connector types are most common for Enovos?"*. These are low-value tasks that consume high-value engineering hours.

**The Solution:** We are building a unified AI Copilot that acts as a **Subject Matter Expert for Engineers** and a **Data Analyst for the Business**. It instantly answers pipeline architecture questions using our exact codebase docs, and queries live data tables directly.

---

## 2. How It Works: The Architecture Flow

When a user submits a query, we don't just dump it into a generic RAG prompt. We use a **deterministic, multi-agent routing system** to guarantee high precision.

*Here is the exact step-by-step flow:*

````mermaid
graph TD
    A[User Submits Query] --> B{Intent Classifier<br/>Model: Claude Sonnet 4.6}
    
    B -->|Intent: STRUCTURED_QUERY| C[Databricks Genie Space]
    B -->|Intent: KNOWLEDGE_LOOKUP| D[Metadata Extractor<br/>Model: Claude Sonnet 4.6]
    B -->|Intent: HYBRID| C
    B -->|Intent: HYBRID| D
    
    %% Genie Path
    C -->|Translates to SQL| J[(Live EUH Tables)]
    J --> K[Format Table Data]
    
    %% Knowledge Path
    D -->|Extracts: Source, Layer, Section| E{Structured SQL Retrieval}
    E -->|Lookup in Delta Table| F[(copilot_knowledge_chunks)]
    
    F -->|If Match Found| G[Top Chunks Retrieved]
    F -->|If No Match Found| H[Vector Search Fallback<br/>Model: GTE-Large-En]
    H -->|Semantic Search| I[(Databricks Vector Index)]
    I --> G
    
    %% Synthesis
    G --> L[Response Composer<br/>Model: Claude Sonnet 4.6]
    L -->|Synthesizes Answer + Citations| M[Final Output & UI]
    K --> M
````

### **Step-by-Step Execution:**

1. **Intent Classification (`databricks-claude-sonnet-4-6`)**: 
   The moment a query hits, an ultra-fast LLM call decides if the user wants code/architecture knowledge (`KNOWLEDGE`), hard data metrics (`STRUCTURED_DATA`), or both (`HYBRID`).
2. **Knowledge Route - Structured-First RAG (`databricks-claude-sonnet-4-6`)**: 
   Instead of unreliable fuzzy searches, we do an LLM extraction (e.g., pulling "Spirii", "EUH Layer", "Business Rules"). We run a direct SQL `WHERE` clause against our Delta table of documentation. **This guarantees 100% accuracy for specific technical questions and costs $0 in compute.**
3. **Knowledge Route - Vector Fallback (`databricks-gte-large-en`)**: 
   If the SQL lookup fails (vague query), we fall back to a Mosaic AI Vector Search index using GTE-Large embeddings to find semantic matches.
4. **Data Route - Genie Space**: 
   If it's a data question, the query routes to a Databricks Genie Space hooked up to our curated reporting views (`v_genie_charger_locations`, etc.), converting English to SQL in real-time.
5. **Synthesis (`databricks-claude-sonnet-4-6`)**: 
   The Response Composer LLM looks at the retrieved documentation chunks (or data), formats a professional response, and appends explicit citations (e.g., `[Source: driivz §deduplication_logic]`).

---

## 3. Why Mosaic AI Vector Search is a Game Changer Here

When our structured SQL lookup doesn't find an exact metadata match, we lean heavily on Databricks Vector Search. Here is exactly how it works and why this specific implementation is a massive upgrade over traditional search:

**How it works under the hood:**
Every chunk of our pipeline documentation is passed through the `databricks-gte-large-en` embedding model, converting the text into a 1024-dimensional mathematical vector. When a user asks a vague question (e.g., *"How do we handle duplicate EVSE records?"*), the query is also converted into a vector. The Vector Search engine calculates the mathematical distance (Cosine Similarity) between the query vector and all the doc vectors, instantly returning the closest conceptual matches.

**Why it's a game changer for our platform:**
1. **Zero-Maintenance "Delta Sync":** We aren't manually managing a vector database. We set up a "TRIGGERED" Delta Sync index. When our ingestion pipeline updates the `copilot_knowledge_chunks` Delta table with new markdown docs, the Vector Index automatically syncs the embeddings.
2. **Semantic Understanding, Not Keyword Matching:** A user can ask *"how do we clean up repeating data?"* and the Vector Search will successfully return the **Deduplication Logic** module. It understands *meaning*, not just raw strings.
3. **Hybrid Search (Pre-filtering):** We push down structured metadata filters into the Vector Search. If the LLM knows the user is asking about "Spirii", the Vector Search explicitly filters for `source_name = 'spirii'` *before* doing the math. This prevents hallucinated cross-contamination between different pipeline architectures.

---

## 4. Cost-Benefit Analysis

### **The Costs (Monthly Estimate)**
We architected this to be notoriously cheap by avoiding heavy LLM usage where standard code (SQL) works perfectly.

- **Claude Sonnet 4.6 Inferences:** ~$10 - $30/month (assuming ~1000 queries. The metadata extraction only uses ~200 tokens).
- **GTE-Large Embeddings:** < $1/month (Only triggered on fallback).
- **Mosaic AI Vector Search Endpoint:** ~$150 - $360/month. *(This is the only fixed infrastructure cost. It runs 24/7. Because our Structured RAG is so effective, we could potentially deprecate this endpoint entirely to save costs if we decide fuzzy search isn't needed).*

**Total TCO:** ~$160 - $390 / month.

### **The Immediate Value (ROI)**
- **Reclaimed Engineering Hours:** If this saves just **3 engineering hours a month** globally (through deflected ad-hoc queries and faster pipeline debugging), it pays for itself.
- **Onboarding Velocity:** New hires can ask the Copilot *"How is the session_duration column calculated for Driivz raw?"* and get an immediate, cited answer instead of digging through Git history or reverse-engineering notebooks.
- **Documentation Enforcement:** This architecture forces a standardized documentation pattern. If engineers want their code searchable by the Copilot, they must adhere to our Markdown templates.
