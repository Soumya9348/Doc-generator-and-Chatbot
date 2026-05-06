# Databricks notebook source
# MAGIC %md
# MAGIC # 🧪 Test Copilot App Locally
# MAGIC
# MAGIC This notebook starts the Flask app on the cluster driver so you can test the premium UI
# MAGIC before deploying as a Databricks App.

# COMMAND ----------

# MAGIC %pip install flask>=3.0 plotly>=5.0 databricks-sql-connector>=3.0 databricks-vectorsearch>=0.40
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure Environment

# COMMAND ----------

import os

# Get Databricks context for proxy URL and SQL connector
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
tags = ctx.tags()

host_name = tags.get("browserHostName").get()
org_id = tags.get("orgId").get()
cluster_id = tags.get("clusterId").get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# SQL Warehouse — using the cluster's own SQL endpoint
# If you have a SQL Warehouse, replace the http_path below
http_path = f"/sql/1.0/warehouses/{cluster_id}"  # or your SQL Warehouse path

# Set environment variables for the orchestrator
os.environ["DATABRICKS_SERVER_HOSTNAME"] = host_name
os.environ["DATABRICKS_HTTP_PATH"] = f"/sql/protocolv1/o/{org_id}/{cluster_id}"
os.environ["DATABRICKS_TOKEN"] = token

# ─── UPDATE THIS with your actual Genie Space ID ───
os.environ["GENIE_SPACE_ID"] = "PUT_YOUR_GENIE_SPACE_ID_HERE"

print("✅ Environment configured")
print(f"   Host: {host_name}")
print(f"   Cluster: {cluster_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Patch orchestrator for cluster testing
# MAGIC
# MAGIC On a cluster, we have `spark` available directly. Let's patch the SQL runner
# MAGIC to use Spark instead of databricks-sql-connector for simpler testing.

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Users/soumya.sourav.behera@2goenergy.com/copilot_app")
# ↑ UPDATE this path to match where you uploaded copilot_app/ in your Workspace

# Patch: Override run_sql to use Spark (available on cluster, simpler for testing)
import orchestrator

def run_sql_spark(query: str) -> list[dict]:
    """Use Spark SQL instead of databricks-sql-connector for cluster testing."""
    try:
        rows = spark.sql(query).collect()
        if not rows:
            return []
        columns = rows[0].asDict().keys()
        return [row.asDict() for row in rows]
    except Exception as e:
        print(f"SQL error: {e}")
        return []

# Replace the orchestrator's run_sql with our Spark-based version
orchestrator.run_sql = run_sql_spark
print("✅ Orchestrator loaded & patched for cluster testing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Quick test — verify orchestrator works

# COMMAND ----------

copilot = orchestrator.CopilotOrchestrator()
result = copilot.query("What are the business rules for Spirii EUH pipeline?")

print(f"\n✅ Orchestrator test passed!")
print(f"   Intent: {result['intent']}")
print(f"   Method: {result['retrieval_method']}")
print(f"   Latency: {result['latency_ms']}ms")
print(f"   Answer preview: {result['answer'][:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Start Flask App

# COMMAND ----------

import threading
from flask import Flask, request, jsonify, send_from_directory

# ─── Build Flask app inline (same as app.py but adapted for notebook) ───

app = Flask(__name__, static_folder="/Workspace/Users/soumya.sourav.behera@2goenergy.com/copilot_app/static")
# ↑ UPDATE this path too

sessions = {}

def get_orchestrator(conversation_id=None):
    if conversation_id and conversation_id in sessions:
        return sessions[conversation_id]
    c = orchestrator.CopilotOrchestrator()
    sessions[c.conversation_id] = c
    if len(sessions) > 50:
        del sessions[list(sessions.keys())[0]]
    return c

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/query", methods=["POST"])
def api_query():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        conversation_id = data.get("conversation_id")
        if not message:
            return jsonify({"error": "Empty message"}), 400

        cop = get_orchestrator(conversation_id)
        result = cop.query(message)

        return jsonify({
            "answer": result.get("answer", ""),
            "intent": result.get("intent", "UNKNOWN"),
            "retrieval_method": result.get("retrieval_method", ""),
            "latency_ms": result.get("latency_ms", 0),
            "citations": result.get("citations", []),
            "sql_generated": result.get("sql_generated"),
            "chart_data": result.get("chart_data"),
            "conversation_id": result.get("conversation_id", cop.conversation_id),
            "turn_number": result.get("turn_number", 0),
        })
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    try:
        data = request.get_json()
        feedback = data.get("feedback", "")
        # For testing, just log it
        print(f"📝 Feedback received: {feedback}")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})

# ─── Start Flask in a background thread ───
PORT = 8000

def run_flask():
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

import time
time.sleep(2)  # Wait for Flask to start

# ─── Print the access URL ───
proxy_url = f"https://{host_name}/driver-proxy/o/{org_id}/{cluster_id}/{PORT}/"

print("\n" + "=" * 80)
print("🚀 COPILOT APP IS RUNNING!")
print(f"👉 {proxy_url}")
print("=" * 80)
print(f"\nOpen the URL above in your browser to test the premium UI.")
print("The app will keep running as long as this notebook cell is active.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🛑 Stop the App
# MAGIC
# MAGIC The app runs in a background thread. To stop it:
# MAGIC - **Detach** this notebook from the cluster, OR
# MAGIC - **Cancel** the cell above, OR
# MAGIC - **Restart** the cluster
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Troubleshooting
# MAGIC
# MAGIC | Issue | Fix |
# MAGIC |-------|-----|
# MAGIC | "Connection refused" | Make sure the cell above is still running. Wait 5 seconds after it prints the URL |
# MAGIC | Port 8000 already in use | Change `PORT = 8000` to another port (e.g., 8001) and update the URL accordingly |
# MAGIC | "Module not found: orchestrator" | Update the `sys.path.insert()` path in Step 2 to match your workspace path |
# MAGIC | Static files not loading | Update the `static_folder` path in Step 4 to match your workspace path |
# MAGIC | Orchestrator errors | Run Step 3 first to verify the orchestrator works independently |
