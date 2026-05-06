"""
eMobility Copilot — Flask Backend
Serves the premium chat UI and exposes the orchestrator via REST API.
Deployed as a Databricks App (no Model Serving Endpoint needed).
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory

from orchestrator import CopilotOrchestrator

# ─────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")

# Store orchestrator instances per session (simple in-memory)
sessions: dict[str, CopilotOrchestrator] = {}

def get_orchestrator(conversation_id: str = None) -> CopilotOrchestrator:
    """Get or create an orchestrator instance for a conversation."""
    if conversation_id and conversation_id in sessions:
        return sessions[conversation_id]
    copilot = CopilotOrchestrator()
    sessions[copilot.conversation_id] = copilot
    # Keep only last 50 sessions to prevent memory leak
    if len(sessions) > 50:
        oldest = list(sessions.keys())[0]
        del sessions[oldest]
    return copilot


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the premium chat UI."""
    return send_from_directory("static", "index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    """Process a user query through the orchestrator."""
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        conversation_id = data.get("conversation_id")

        if not message:
            return jsonify({"error": "Empty message"}), 400

        copilot = get_orchestrator(conversation_id)
        result = copilot.query(message)

        return jsonify({
            "answer": result.get("answer", ""),
            "intent": result.get("intent", "UNKNOWN"),
            "retrieval_method": result.get("retrieval_method", ""),
            "latency_ms": result.get("latency_ms", 0),
            "citations": result.get("citations", []),
            "sql_generated": result.get("sql_generated"),
            "chart_data": result.get("chart_data"),
            "conversation_id": result.get("conversation_id", copilot.conversation_id),
            "turn_number": result.get("turn_number", 0),
        })

    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Log user feedback (thumbs up/down)."""
    try:
        data = request.get_json()
        conversation_id = data.get("conversation_id", "")
        turn_number = data.get("turn_number", 0)
        feedback = data.get("feedback", "")  # "positive" or "negative"

        from orchestrator import run_sql, CONFIG
        run_sql(f"""
            UPDATE {CONFIG['conversations_table']}
            SET feedback = '{feedback}'
            WHERE conversation_id = '{conversation_id}'
              AND turn_number = {turn_number}
        """)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy", "app": "eMobility Copilot"})


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 eMobility Copilot starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
