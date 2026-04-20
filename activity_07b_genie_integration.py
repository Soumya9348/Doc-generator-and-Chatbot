# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 Activity 7b: Genie Space Integration
# MAGIC
# MAGIC **Goal**: Create reporting views for Genie Space, create the Space in UI, and test the API integration.
# MAGIC
# MAGIC **Steps**:
# MAGIC 1. Explore what EUH tables exist in your Dev catalog
# MAGIC 2. Create reporting views for Genie
# MAGIC 3. Create Genie Space in the UI (manual step)
# MAGIC 4. Test the Genie API from code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Explore Available EUH Tables
# MAGIC
# MAGIC First, let's see what tables exist that Genie can query.
# MAGIC Run this cell and share the output — I'll help create views based on what's available.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- List all tables/views in the sandbox schema
# MAGIC SHOW TABLES IN `emobility-uc-dev`.`sandbox-emobility`;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- List tables in the EUH schema (this is where the transformed data lives)
# MAGIC -- UPDATE the schema name if different
# MAGIC SHOW TABLES IN `emobility-uc-dev`.`euh`;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore EUH Table Schemas
# MAGIC
# MAGIC Uncomment the tables that exist in your workspace.
# MAGIC Based on the ingested docs, possible EUH tables are:
# MAGIC - charger_location
# MAGIC - charger_evse  
# MAGIC - charger_connector
# MAGIC - charger_session
# MAGIC - charger_event

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Uncomment each line to check which tables exist and see their columns
# MAGIC -- DESCRIBE TABLE `emobility-uc-dev`.`euh`.charger_location;
# MAGIC -- DESCRIBE TABLE `emobility-uc-dev`.`euh`.charger_evse;
# MAGIC -- DESCRIBE TABLE `emobility-uc-dev`.`euh`.charger_connector;
# MAGIC -- DESCRIBE TABLE `emobility-uc-dev`.`euh`.charger_session;
# MAGIC -- DESCRIBE TABLE `emobility-uc-dev`.`euh`.charger_event;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Quick row counts for the tables that exist
# MAGIC -- Uncomment as needed
# MAGIC -- SELECT 'charger_location' as tbl, COUNT(*) as rows FROM `emobility-uc-dev`.`euh`.charger_location
# MAGIC -- UNION ALL SELECT 'charger_evse', COUNT(*) FROM `emobility-uc-dev`.`euh`.charger_evse
# MAGIC -- UNION ALL SELECT 'charger_connector', COUNT(*) FROM `emobility-uc-dev`.`euh`.charger_connector
# MAGIC -- UNION ALL SELECT 'charger_session', COUNT(*) FROM `emobility-uc-dev`.`euh`.charger_session
# MAGIC -- UNION ALL SELECT 'charger_event', COUNT(*) FROM `emobility-uc-dev`.`euh`.charger_event;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Create Reporting Views for Genie
# MAGIC
# MAGIC These views add **business-friendly column descriptions** so Genie understands what each column means.
# MAGIC Without descriptions, Genie can't answer natural language questions accurately.
# MAGIC
# MAGIC **📌 Update table/schema names below based on Step 1 output.**

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ═════════════════════════════════════════════════════════════════
# MAGIC -- VIEW 1: Charging Locations (for questions like "how many stations?")
# MAGIC -- ═════════════════════════════════════════════════════════════════
# MAGIC CREATE OR REPLACE VIEW `emobility-uc-dev`.`sandbox-emobility`.v_genie_charger_locations
# MAGIC (
# MAGIC   source            COMMENT 'Source system the data came from (driivz, spirii, uberall, enovos)',
# MAGIC   source_location_id COMMENT 'Unique ID of the charging location in the source system',
# MAGIC   location_name     COMMENT 'Human-readable name of the charging station/location',
# MAGIC   operator          COMMENT 'The operator/company managing this charging location',
# MAGIC   owning_company    COMMENT 'The company that owns this charging location',
# MAGIC   country_code      COMMENT 'ISO country code where the location is situated (e.g., DE, NL, DK)',
# MAGIC   city              COMMENT 'City where the charging location is situated',
# MAGIC   address           COMMENT 'Street address of the charging location',
# MAGIC   postal_code       COMMENT 'Postal/ZIP code of the charging location',
# MAGIC   latitude          COMMENT 'GPS latitude coordinate of the location',
# MAGIC   longitude         COMMENT 'GPS longitude coordinate of the location',
# MAGIC   status            COMMENT 'Current status of the location (e.g., active, inactive)',
# MAGIC   location_type     COMMENT 'Type of the charging location (e.g., Shell Recharge)',
# MAGIC   created           COMMENT 'Timestamp when this location record was first created'
# MAGIC )
# MAGIC AS
# MAGIC SELECT
# MAGIC   source,
# MAGIC   source_location_id,
# MAGIC   name AS location_name,
# MAGIC   operator,
# MAGIC   owning_company,
# MAGIC   country_code,
# MAGIC   city,
# MAGIC   address,
# MAGIC   postal_code,
# MAGIC   latitude,
# MAGIC   longitude,
# MAGIC   status,
# MAGIC   location_type,
# MAGIC   created
# MAGIC FROM `emobility-uc-dev`.`euh`.charger_location;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ═════════════════════════════════════════════════════════════════
# MAGIC -- VIEW 2: Charging Points / EVSE (for "how many chargers?" questions)
# MAGIC -- ═════════════════════════════════════════════════════════════════
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
# MAGIC   modified          COMMENT 'When this EVSE record was last modified'
# MAGIC )
# MAGIC AS
# MAGIC SELECT
# MAGIC   source,
# MAGIC   location_id,
# MAGIC   source_location_id,
# MAGIC   source_evse_id,
# MAGIC   chargepoint_id,
# MAGIC   latitude,
# MAGIC   longitude,
# MAGIC   created,
# MAGIC   modified
# MAGIC FROM `emobility-uc-dev`.`euh`.charger_evse;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ═════════════════════════════════════════════════════════════════
# MAGIC -- VIEW 3: Connectors (for "connector types?" / "power distribution?")
# MAGIC -- ═════════════════════════════════════════════════════════════════
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
# MAGIC SELECT
# MAGIC   source,
# MAGIC   source_connector_id,
# MAGIC   source_evse_id,
# MAGIC   connector_type,
# MAGIC   power_type,
# MAGIC   power_kw,
# MAGIC   phase
# MAGIC FROM `emobility-uc-dev`.`euh`.charger_connector;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ═════════════════════════════════════════════════════════════════
# MAGIC -- VIEW 4: Charging Sessions (if table exists — driivz has this)
# MAGIC -- ═════════════════════════════════════════════════════════════════
# MAGIC -- UNCOMMENT if charger_session table exists in your dev environment
# MAGIC
# MAGIC -- CREATE OR REPLACE VIEW `emobility-uc-dev`.`sandbox-emobility`.v_genie_charger_sessions
# MAGIC -- (
# MAGIC --   source                COMMENT 'Source system (driivz)',
# MAGIC --   source_session_id     COMMENT 'Unique session ID in the source system',
# MAGIC --   location_id           COMMENT 'Location where the session occurred',
# MAGIC --   evse_id               COMMENT 'EVSE used for this session',
# MAGIC --   connector_id          COMMENT 'Connector used for this session',
# MAGIC --   session_start         COMMENT 'Timestamp when the charging session started',
# MAGIC --   session_end           COMMENT 'Timestamp when the charging session ended',
# MAGIC --   session_duration_seconds COMMENT 'Total session duration in seconds (includes idle time)',
# MAGIC --   charging_duration_seconds COMMENT 'Actual charging duration in seconds (energy flowing)',
# MAGIC --   energy_kwh            COMMENT 'Total energy delivered in kilowatt-hours (kWh)',
# MAGIC --   status                COMMENT 'Session status (completed, failed, etc.)'
# MAGIC -- )
# MAGIC -- AS
# MAGIC -- SELECT
# MAGIC --   source,
# MAGIC --   source_session_id,
# MAGIC --   location_id,
# MAGIC --   evse_id,
# MAGIC --   connector_id,
# MAGIC --   session_start,
# MAGIC --   session_end,
# MAGIC --   session_duration_seconds,
# MAGIC --   charging_duration_seconds,
# MAGIC --   energy_kwh,
# MAGIC --   status
# MAGIC -- FROM `emobility-uc-dev`.`euh`.charger_session;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify views were created
# MAGIC SHOW VIEWS IN `emobility-uc-dev`.`sandbox-emobility` LIKE 'v_genie_*';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Quick test: sample data from the location view
# MAGIC SELECT * FROM `emobility-uc-dev`.`sandbox-emobility`.v_genie_charger_locations LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Create Genie Space in UI
# MAGIC
# MAGIC **This is a manual step in the Databricks UI.**
# MAGIC
# MAGIC ### Instructions:
# MAGIC
# MAGIC 1. Go to **Databricks Workspace** → left sidebar → **Genie** (under AI section)
# MAGIC 2. Click **"New"** to create a new Genie Space
# MAGIC 3. **Name**: `eMobility Copilot - Data Assistant`
# MAGIC 4. **Description**: `Ask questions about EV charging stations, connectors, EVSE devices, and charging sessions across all source systems (Driivz, Spirii, Uberall, Enovos).`
# MAGIC 5. **Add tables/views** — select these views from `emobility-uc-dev.sandbox-emobility`:
# MAGIC    - `v_genie_charger_locations`
# MAGIC    - `v_genie_charger_evse`
# MAGIC    - `v_genie_charger_connectors`
# MAGIC    - `v_genie_charger_sessions` (if created)
# MAGIC 6. **SQL Warehouse**: Select your available SQL warehouse
# MAGIC 7. Click **Create**
# MAGIC
# MAGIC ### Get the Space ID:
# MAGIC After creating, look at the URL — it will contain the Space ID:
# MAGIC ```
# MAGIC https://<workspace>.cloud.databricks.com/genie/rooms/<SPACE_ID>
# MAGIC ```
# MAGIC Copy that `<SPACE_ID>` and paste it below.

# COMMAND ----------

# ═════════════════════════════════════════════════════════════════
# PASTE YOUR GENIE SPACE ID HERE
# ═════════════════════════════════════════════════════════════════
GENIE_SPACE_ID = "PUT_YOUR_GENIE_SPACE_ID_HERE"

print(f"Genie Space ID: {GENIE_SPACE_ID}")
if GENIE_SPACE_ID == "PUT_YOUR_GENIE_SPACE_ID_HERE":
    print("⚠️  Please update the ID above after creating the Space in the UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Test Genie API

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

def test_genie_query(space_id: str, question: str, timeout: int = 120) -> dict:
    """
    Send a question to Genie Space and return the answer.
    """
    w = WorkspaceClient()
    
    print(f"💬 Sending to Genie: \"{question}\"")
    
    # Start conversation
    response = w.genie.start_conversation(
        space_id=space_id,
        content=question,
    )
    
    conversation_id = response.conversation_id
    message_id = response.message_id
    print(f"   Conversation: {conversation_id}")
    print(f"   Message:      {message_id}")
    
    # Poll for result
    waited = 0
    while waited < timeout:
        msg = w.genie.get_message(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        
        status = msg.status.value if hasattr(msg.status, 'value') else str(msg.status)
        
        if status in ("COMPLETED", "COMPLETED_WITH_ERROR"):
            print(f"   ✅ Status: {status} ({waited}s)")
            
            # Extract results
            answer = ""
            sql = None
            
            for attachment in (msg.attachments or []):
                if hasattr(attachment, 'text') and attachment.text:
                    answer += attachment.text.content + "\n"
                
                if hasattr(attachment, 'query') and attachment.query:
                    sql = attachment.query.query
                    print(f"\n   📊 SQL Generated:\n   {sql}")
                    
                    # Format results as table
                    if hasattr(attachment.query, 'result') and attachment.query.result:
                        result = attachment.query.result
                        columns = [col.name for col in (result.columns or [])]
                        data = result.data_array or []
                        
                        if columns and data:
                            answer += "\n| " + " | ".join(columns) + " |\n"
                            answer += "| " + " | ".join(["---"] * len(columns)) + " |\n"
                            for row in data[:20]:
                                answer += "| " + " | ".join(str(v or "") for v in row) + " |\n"
            
            return {"answer": answer.strip() or "Query completed but no text returned.",
                    "sql": sql, "status": status}
        
        elif status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
            print(f"   ❌ Status: {status}")
            return {"answer": f"Genie query failed: {status}", "sql": None, "status": status}
        
        else:
            if waited % 10 == 0:
                print(f"   ⏳ Status: {status} — waiting... ({waited}s)")
            time.sleep(3)
            waited += 3
    
    return {"answer": "Genie query timed out.", "sql": None, "status": "TIMEOUT"}


if GENIE_SPACE_ID != "PUT_YOUR_GENIE_SPACE_ID_HERE":
    print("✅ Genie API test function ready")
else:
    print("⚠️  Set GENIE_SPACE_ID first, then run the test cells below")

# COMMAND ----------

# ─── Test Query 1 ───
if GENIE_SPACE_ID != "PUT_YOUR_GENIE_SPACE_ID_HERE":
    result = test_genie_query(GENIE_SPACE_ID, "How many charging locations do we have by country?")
    print(f"\n📝 Answer:\n{result['answer']}")

# COMMAND ----------

# ─── Test Query 2 ───
if GENIE_SPACE_ID != "PUT_YOUR_GENIE_SPACE_ID_HERE":
    result = test_genie_query(GENIE_SPACE_ID, "What connector types do we have and how many of each?")
    print(f"\n📝 Answer:\n{result['answer']}")

# COMMAND ----------

# ─── Test Query 3 ───
if GENIE_SPACE_ID != "PUT_YOUR_GENIE_SPACE_ID_HERE":
    result = test_genie_query(GENIE_SPACE_ID, "Show me the top 5 cities with the most charging stations")
    print(f"\n📝 Answer:\n{result['answer']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Update Orchestrator with Genie Space ID
# MAGIC
# MAGIC Once Genie is working, update the `CONFIG["genie_space_id"]` in the Activity 7 orchestrator notebook.
# MAGIC
# MAGIC ```python
# MAGIC CONFIG = {
# MAGIC     ...
# MAGIC     "genie_space_id": "<YOUR_ACTUAL_SPACE_ID>",
# MAGIC     ...
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Then re-run the orchestrator to test end-to-end:
# MAGIC ```python
# MAGIC copilot.query("How many charging locations do we have in Germany?")  # → routes to Genie
# MAGIC copilot.query("What are the business rules for Spirii?")            # → routes to Knowledge Agent
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Activity 7b Complete
# MAGIC
# MAGIC **Created:**
# MAGIC - `v_genie_charger_locations` — locations with business-friendly column descriptions
# MAGIC - `v_genie_charger_evse` — charging points/EVSE devices
# MAGIC - `v_genie_charger_connectors` — connector types and power specs
# MAGIC - `v_genie_charger_sessions` — (template, uncomment if table exists)
# MAGIC - Genie API integration tested
# MAGIC
# MAGIC **Next:**
# MAGIC 1. Create the Genie Space in the UI (Step 3 instructions above)
# MAGIC 2. Paste the Space ID
# MAGIC 3. Run the test queries
# MAGIC 4. Update the orchestrator with the Space ID
# MAGIC 5. → **Activity 8: Chat UI (Gradio)**
