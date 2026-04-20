# Databricks notebook source
# MAGIC %md
# MAGIC # 💎 Activity 8: Copilot Chat UI
# MAGIC
# MAGIC Premium Gradio chat interface with:
# MAGIC - Dark glassmorphism theme
# MAGIC - Markdown-rendered responses (tables, code blocks, headers)
# MAGIC - Interactive Plotly charts for data queries
# MAGIC - Citations panel with source tracking
# MAGIC - Intent & retrieval method badges
# MAGIC - Quick suggestion chips
# MAGIC - Feedback system (thumbs up/down)
# MAGIC - Latency & cost metrics

# COMMAND ----------

# MAGIC %pip install gradio>=4.0 plotly
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Orchestrator
# MAGIC
# MAGIC Run Activity 7 to import all functions.

# COMMAND ----------

# MAGIC %run ./activity_07_orchestrator

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the UI

# COMMAND ----------

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re
import time

# ─────────────────────────────────────────────────────────────────
# Premium CSS — Dark Glassmorphism Theme
# ─────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ═══════ GLOBAL ═══════ */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%) !important;
    min-height: 100vh;
}
.main {
    background: transparent !important;
}

/* ═══════ HEADER ═══════ */
#header-row {
    background: linear-gradient(135deg, rgba(79,139,249,0.15), rgba(124,58,237,0.15));
    border: 1px solid rgba(79,139,249,0.2);
    border-radius: 16px;
    padding: 20px 28px !important;
    backdrop-filter: blur(20px);
    margin-bottom: 12px;
}
#header-row * {
    color: #e2e8f0 !important;
}

/* ═══════ CHATBOT ═══════ */
#copilot-chatbot {
    background: rgba(15,17,23,0.85) !important;
    border: 1px solid rgba(79,139,249,0.15) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(20px);
    min-height: 520px !important;
}
#copilot-chatbot .message {
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin: 6px 0 !important;
    line-height: 1.6;
    font-size: 14px;
}
#copilot-chatbot .message.user {
    background: linear-gradient(135deg, rgba(79,139,249,0.25), rgba(79,139,249,0.1)) !important;
    border: 1px solid rgba(79,139,249,0.3) !important;
    color: #e2e8f0 !important;
}
#copilot-chatbot .message.bot {
    background: rgba(30,33,48,0.9) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: #cbd5e1 !important;
}
#copilot-chatbot .message.bot table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 13px;
}
#copilot-chatbot .message.bot th {
    background: rgba(79,139,249,0.2);
    color: #93c5fd;
    padding: 8px 12px;
    text-align: left;
    border: 1px solid rgba(79,139,249,0.2);
}
#copilot-chatbot .message.bot td {
    padding: 6px 12px;
    border: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e1;
}
#copilot-chatbot .message.bot tr:nth-child(even) {
    background: rgba(79,139,249,0.05);
}
#copilot-chatbot .message.bot code {
    background: rgba(79,139,249,0.15);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
    color: #93c5fd;
}
#copilot-chatbot .message.bot pre {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(79,139,249,0.15);
    border-radius: 8px;
    padding: 12px;
    overflow-x: auto;
}
#copilot-chatbot .message.bot h1, 
#copilot-chatbot .message.bot h2, 
#copilot-chatbot .message.bot h3 {
    color: #93c5fd !important;
    margin: 12px 0 6px 0;
}
#copilot-chatbot .message.bot strong {
    color: #e2e8f0 !important;
}

/* ═══════ INPUT ═══════ */
#user-input textarea {
    background: rgba(30,33,48,0.9) !important;
    border: 1px solid rgba(79,139,249,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
    transition: all 0.3s ease;
}
#user-input textarea:focus {
    border-color: rgba(79,139,249,0.6) !important;
    box-shadow: 0 0 20px rgba(79,139,249,0.15) !important;
}
#user-input textarea::placeholder {
    color: #64748b !important;
}

/* ═══════ BUTTONS ═══════ */
#send-btn {
    background: linear-gradient(135deg, #4f8bf9, #7c3aed) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(79,139,249,0.3);
}
#send-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(79,139,249,0.4) !important;
}
.suggestion-btn {
    background: rgba(30,33,48,0.8) !important;
    border: 1px solid rgba(79,139,249,0.2) !important;
    border-radius: 20px !important;
    color: #93c5fd !important;
    font-size: 12px !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease;
    cursor: pointer;
}
.suggestion-btn:hover {
    background: rgba(79,139,249,0.15) !important;
    border-color: rgba(79,139,249,0.4) !important;
    transform: translateY(-1px);
}
#feedback-up, #feedback-down {
    border-radius: 10px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    transition: all 0.3s ease;
}
#feedback-up {
    background: rgba(34,197,94,0.15) !important;
    border: 1px solid rgba(34,197,94,0.3) !important;
    color: #4ade80 !important;
}
#feedback-up:hover {
    background: rgba(34,197,94,0.25) !important;
}
#feedback-down {
    background: rgba(239,68,68,0.15) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    color: #f87171 !important;
}
#feedback-down:hover {
    background: rgba(239,68,68,0.25) !important;
}
#clear-btn {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    border-radius: 10px !important;
    color: #f87171 !important;
}

/* ═══════ METADATA PANEL ═══════ */
#metadata-panel, #citations-panel, #chart-panel {
    background: rgba(15,17,23,0.85) !important;
    border: 1px solid rgba(79,139,249,0.12) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(20px);
}
#metadata-panel .label-wrap,
#citations-panel .label-wrap,
#chart-panel .label-wrap {
    color: #93c5fd !important;
}

/* ═══════ METRICS BADGES ═══════ */
.metric-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px 4px;
}
.badge-structured {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1));
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
}
.badge-vector {
    background: linear-gradient(135deg, rgba(234,179,8,0.2), rgba(234,179,8,0.1));
    color: #facc15;
    border: 1px solid rgba(234,179,8,0.3);
}
.badge-hybrid {
    background: linear-gradient(135deg, rgba(79,139,249,0.2), rgba(124,58,237,0.1));
    color: #93c5fd;
    border: 1px solid rgba(79,139,249,0.3);
}
.badge-genie {
    background: linear-gradient(135deg, rgba(168,85,247,0.2), rgba(168,85,247,0.1));
    color: #c084fc;
    border: 1px solid rgba(168,85,247,0.3);
}
.badge-knowledge {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1));
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
}
.badge-data {
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(59,130,246,0.1));
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.3);
}

/* ═══════ SCROLLBAR ═══════ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(79,139,249,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(79,139,249,0.5); }

/* ═══════ LABELS & TEXT ═══════ */
label, .label-wrap span {
    color: #94a3b8 !important;
    font-size: 13px !important;
}
.block {
    background: transparent !important;
    border: none !important;
}
"""

# ─────────────────────────────────────────────────────────────────
# Suggestion Queries
# ─────────────────────────────────────────────────────────────────

SUGGESTIONS = {
    "📚 Knowledge": [
        "What are the business rules for Spirii EUH pipeline?",
        "How is the power_kw column derived in Spirii?",
        "What deduplication logic does Driivz use in raw layer?",
        "What is Enovos and what data does it provide?",
    ],
    "📊 Data": [
        "How many charging locations by country?",
        "Show top 5 cities with most charging stations",
        "What connector types do we have?",
        "How many EVSE devices per source system?",
    ],
}

# ─────────────────────────────────────────────────────────────────
# Response Processing Helpers
# ─────────────────────────────────────────────────────────────────

def format_metadata_html(result: dict) -> str:
    """Generate rich HTML metadata display."""
    intent = result.get("intent", "unknown")
    method = result.get("retrieval_method", "unknown")
    latency = result.get("latency_ms", 0)

    # Intent badge
    intent_class = "badge-knowledge" if intent == "KNOWLEDGE_LOOKUP" else ("badge-data" if intent == "STRUCTURED_QUERY" else "badge-hybrid")
    intent_label = {"KNOWLEDGE_LOOKUP": "📚 Knowledge", "STRUCTURED_QUERY": "📊 Data Query", "HYBRID": "🔗 Hybrid"}.get(intent, intent)

    # Method badge
    method_class = f"badge-{method}"
    method_label = {"structured": "🗄️ Structured", "vector": "🔍 Vector", "hybrid": "🔗 Hybrid", "genie": "✨ Genie"}.get(method, method)

    # Latency color
    lat_color = "#4ade80" if latency < 3000 else ("#facc15" if latency < 8000 else "#f87171")

    html = f"""
<div style="padding: 8px 0;">
    <div style="margin-bottom: 10px;">
        <span class="metric-badge {intent_class}">{intent_label}</span>
        <span class="metric-badge {method_class}">{method_label}</span>
    </div>
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:8px;">
        <div style="color:#94a3b8; font-size:12px;">
            ⏱️ <span style="color:{lat_color}; font-weight:600;">{latency/1000:.1f}s</span>
        </div>
        <div style="color:#94a3b8; font-size:12px;">
            📦 <span style="color:#e2e8f0; font-weight:600;">{result.get('chunks_used', len(result.get('citations', [])))} sources</span>
        </div>
        <div style="color:#94a3b8; font-size:12px;">
            🔄 Turn <span style="color:#e2e8f0; font-weight:600;">#{result.get('turn_number', '?')}</span>
        </div>
    </div>
</div>"""
    return html


def format_citations_html(result: dict) -> str:
    """Generate rich HTML citations display."""
    citations = result.get("citations", [])
    if not citations:
        return "<div style='color:#64748b; padding:8px; font-style:italic;'>No citations for this response.</div>"

    html = "<div style='padding:8px 0;'>"
    for i, c in enumerate(citations):
        method = c.get("method", "")
        method_icon = {"structured": "🗄️", "vector": "🔍", "genie": "✨"}.get(method, "📄")
        layer = c.get("layer", "")
        layer_badge = f"<span style='background:rgba(79,139,249,0.1); color:#93c5fd; padding:2px 8px; border-radius:10px; font-size:10px; margin-left:6px;'>{layer}</span>" if layer else ""

        html += f"""
        <div style="padding:8px 12px; margin:4px 0; background:rgba(30,33,48,0.6); border-radius:8px; border-left:3px solid rgba(79,139,249,0.4);">
            <div style="color:#e2e8f0; font-size:13px; font-weight:500;">
                {method_icon} {c.get('source', 'Unknown')}{layer_badge}
            </div>
        </div>"""
    html += "</div>"

    # SQL generated
    sql = result.get("sql_generated")
    if sql:
        html += f"""
        <div style="margin-top:10px; padding:10px 12px; background:rgba(0,0,0,0.3); border-radius:8px; border:1px solid rgba(79,139,249,0.1);">
            <div style="color:#93c5fd; font-size:11px; margin-bottom:6px; font-weight:600;">💾 Generated SQL</div>
            <pre style="color:#cbd5e1; font-size:12px; margin:0; white-space:pre-wrap; word-break:break-all;">{sql}</pre>
        </div>"""

    return html


def try_generate_chart(result: dict) -> go.Figure | None:
    """Try to generate a Plotly chart from Genie results."""
    sql = result.get("sql_generated")
    if not sql:
        return None

    try:
        df = spark.sql(sql).toPandas()
        if df.empty or len(df.columns) < 2:
            return None

        # Determine chart type based on data
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = [c for c in cols if c not in num_cols]

        if not num_cols or not cat_cols:
            return None

        x_col = cat_cols[0]
        y_col = num_cols[0]

        # Limit to top 20 for readability
        if len(df) > 20:
            df = df.head(20)

        # Create chart
        fig = px.bar(
            df, x=x_col, y=y_col,
            color_discrete_sequence=["#4f8bf9"],
            title=f"{y_col} by {x_col}",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15,17,23,0.9)",
            plot_bgcolor="rgba(15,17,23,0.9)",
            font=dict(family="Inter, system-ui", color="#94a3b8", size=12),
            title=dict(font=dict(color="#e2e8f0", size=16)),
            xaxis=dict(gridcolor="rgba(79,139,249,0.1)", title_font=dict(color="#93c5fd")),
            yaxis=dict(gridcolor="rgba(79,139,249,0.1)", title_font=dict(color="#93c5fd")),
            margin=dict(l=40, r=20, t=50, b=40),
            height=320,
        )

        fig.update_traces(
            marker=dict(
                color="#4f8bf9",
                line=dict(width=0),
                cornerradius=6,
            ),
            hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y:,.0f}}<extra></extra>"
        )

        return fig
    except Exception as e:
        print(f"   Chart generation skipped: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Build the Gradio App
# ─────────────────────────────────────────────────────────────────

# Store last result for feedback
last_result = {"value": None}

def respond(message: str, chat_history: list) -> tuple:
    """Process user message and return updated chat, metadata, citations, chart."""
    if not message.strip():
        return chat_history, "", "", None

    # Add user message
    chat_history = chat_history + [{"role": "user", "content": message}]

    # Call orchestrator
    result = copilot.query(message, verbose=False)
    last_result["value"] = result

    # Add bot response
    answer = result.get("answer", "Sorry, I couldn't process that query.")
    chat_history = chat_history + [{"role": "assistant", "content": answer}]

    # Generate metadata & citations HTML
    meta_html = format_metadata_html(result)
    cite_html = format_citations_html(result)

    # Try chart
    chart = try_generate_chart(result)

    return chat_history, meta_html, cite_html, chart


def handle_feedback(feedback_type: str):
    """Log user feedback to conversations table."""
    result = last_result.get("value")
    if result:
        try:
            spark.sql(f"""
                UPDATE {CONFIG['conversations_table']}
                SET feedback = '{feedback_type}'
                WHERE conversation_id = '{result.get("conversation_id", "")}'
                  AND turn_number = {result.get("turn_number", 0)}
            """)
            return f"✅ Feedback recorded: {'👍 Positive' if feedback_type == 'positive' else '👎 Negative'}"
        except Exception as e:
            return f"⚠️ Could not save feedback: {e}"
    return "⚠️ No response to provide feedback on."


def on_suggestion_click(suggestion: str, chat_history: list) -> tuple:
    """Handle suggestion chip click."""
    return respond(suggestion, chat_history) + (suggestion,)


# ─────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    css=CUSTOM_CSS,
    title="eMobility DataPlatform Copilot",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
) as app:

    # ─── Header ───
    with gr.Row(elem_id="header-row"):
        gr.Markdown("""
# ⚡ eMobility DataPlatform Copilot
**Ask about pipelines, transformations, business rules** (Knowledge Agent) **or query your data** (Genie).
Powered by Claude Sonnet 4.6 • Structured-First RAG • Mosaic AI Vector Search
        """)

    # ─── Main Layout ───
    with gr.Row():
        # ─── Chat Panel (75%) ───
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="copilot-chatbot",
                type="messages",
                height=520,
                show_copy_button=True,
                avatar_images=(None, "https://img.icons8.com/fluency/48/bot.png"),
                placeholder="💡 Ask me about your data pipelines, transformations, or query your data...",
            )

            with gr.Row():
                user_input = gr.Textbox(
                    elem_id="user-input",
                    placeholder="Ask about pipelines, business rules, or query your data...",
                    show_label=False,
                    scale=6,
                    lines=1,
                    max_lines=3,
                )
                send_btn = gr.Button("Send ✨", elem_id="send-btn", scale=1, variant="primary")

            # ─── Suggestions ───
            gr.Markdown("### 💡 Quick Questions", elem_classes=["label-wrap"])
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**📚 Knowledge**", elem_classes=["label-wrap"])
                    for s in SUGGESTIONS["📚 Knowledge"]:
                        gr.Button(s, elem_classes=["suggestion-btn"], size="sm").click(
                            fn=lambda s=s: s, outputs=[user_input]
                        )
                with gr.Column():
                    gr.Markdown("**📊 Data**", elem_classes=["label-wrap"])
                    for s in SUGGESTIONS["📊 Data"]:
                        gr.Button(s, elem_classes=["suggestion-btn"], size="sm").click(
                            fn=lambda s=s: s, outputs=[user_input]
                        )

        # ─── Side Panel (25%) ───
        with gr.Column(scale=1):
            with gr.Accordion("📊 Response Metrics", open=True, elem_id="metadata-panel"):
                metadata_display = gr.HTML(
                    value="<div style='color:#64748b; padding:12px; font-style:italic;'>Ask a question to see metrics here.</div>"
                )

            with gr.Accordion("📎 Sources & Citations", open=True, elem_id="citations-panel"):
                citations_display = gr.HTML(
                    value="<div style='color:#64748b; padding:12px; font-style:italic;'>Sources will appear here.</div>"
                )

            with gr.Accordion("📈 Data Visualization", open=True, elem_id="chart-panel"):
                chart_display = gr.Plot(value=None)

            # ─── Feedback ───
            gr.Markdown("### Rate this response", elem_classes=["label-wrap"])
            with gr.Row():
                feedback_up = gr.Button("👍 Helpful", elem_id="feedback-up", size="sm")
                feedback_down = gr.Button("👎 Not helpful", elem_id="feedback-down", size="sm")
            feedback_msg = gr.Markdown("")

            # ─── Clear ───
            clear_btn = gr.Button("🗑️ Clear Chat", elem_id="clear-btn", size="sm")

    # ─── Event Handlers ───
    send_btn.click(
        fn=respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot, metadata_display, citations_display, chart_display],
    ).then(fn=lambda: "", outputs=[user_input])

    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot, metadata_display, citations_display, chart_display],
    ).then(fn=lambda: "", outputs=[user_input])

    feedback_up.click(
        fn=lambda: handle_feedback("positive"),
        outputs=[feedback_msg],
    )
    feedback_down.click(
        fn=lambda: handle_feedback("negative"),
        outputs=[feedback_msg],
    )

    clear_btn.click(
        fn=lambda: ([], 
                    "<div style='color:#64748b; padding:12px; font-style:italic;'>Ask a question to see metrics here.</div>",
                    "<div style='color:#64748b; padding:12px; font-style:italic;'>Sources will appear here.</div>",
                    None,
                    ""),
        outputs=[chatbot, metadata_display, citations_display, chart_display, feedback_msg],
    )


print("✅ Copilot UI built — launching...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Launch

# COMMAND ----------

# Launch the app
# In Databricks, this creates a URL you can access
app.launch(
    share=False,
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Notes
# MAGIC
# MAGIC ### Accessing the UI
# MAGIC
# MAGIC After launching, the app URL will be displayed above. In Databricks:
# MAGIC - The app runs on the cluster driver node
# MAGIC - Access via the **driver proxy URL** shown in the output
# MAGIC - Format: `https://<workspace>.cloud.databricks.com/driver-proxy/o/<org-id>/<cluster-id>/7860/`
# MAGIC
# MAGIC ### If the URL doesn't load:
# MAGIC 1. Try `share=True` in the launch call (creates a public Gradio link — temporary, 72h)
# MAGIC 2. Or deploy as a **Databricks App** for a permanent URL (Activity 10)
# MAGIC
# MAGIC ### Features:
# MAGIC - **Markdown rendering**: Tables, code blocks, headers, bold/italic
# MAGIC - **Charts**: Auto-generated Plotly charts for Genie data queries
# MAGIC - **Citations**: Shows source documents and retrieval method
# MAGIC - **Metrics**: Intent classification, retrieval method, latency
# MAGIC - **Feedback**: Thumbs up/down logged to the conversations table
# MAGIC - **Suggestions**: Quick-click chips for common queries
