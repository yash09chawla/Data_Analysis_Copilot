"""
Data Analysis Copilot — main entry point
Run:  python app.py
Open: http://localhost:7860
"""

import os
import tempfile
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from eda_engine import EDAEngine
from ai_insights import generate_insights, answer_data_question
from report_generator import generate_html_report

load_dotenv()

# ── LLM factory ──────────────────────────────────────────────────────────────
def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. "
            "Get your FREE key at https://console.groq.com → API Keys."
        )
    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",   # best free model on Groq
        temperature=0.2,
        max_tokens=2048,
    )


# ── Global state ──────────────────────────────────────────────────────────────
_current_df: pd.DataFrame | None = None
_df_context_str: str = ""


def _make_df_context(df: pd.DataFrame) -> str:
    """Compact string summary of the dataframe for Q&A context."""
    return f"""Shape: {df.shape}
Columns: {list(df.columns)}
Dtypes:
{df.dtypes.astype(str).to_string()}

Head (3 rows):
{df.head(3).to_string()}

Describe:
{df.describe(include='all').round(3).to_string()}

Missing values:
{df.isnull().sum().to_string()}
"""


# ── Core analysis function ────────────────────────────────────────────────────
def run_analysis(file, target_col_input: str, progress=gr.Progress(track_tqdm=True)):
    global _current_df, _df_context_str

    if file is None:
        return (
            None,
            "⚠ Please upload a CSV file first.",
            None,
            gr.update(value=None),
        )

    try:
        # 1 · Load
        progress(0.05, desc="Loading CSV …")
        df = pd.read_csv(file.name)
        _current_df = df
        _df_context_str = _make_df_context(df)
        dataset_name = os.path.basename(file.name).replace(".csv", "")

        # Detect target column
        target_col = target_col_input.strip() if target_col_input.strip() in df.columns else None

        # 2 · EDA
        progress(0.15, desc="Running EDA engine …")
        engine = EDAEngine(df)
        results = engine.run_all(target_col=target_col)

        # 3 · AI insights via Groq
        progress(0.65, desc="Generating AI insights with Groq LLaMA 3.3 70B …")
        llm     = _get_llm()
        insights = generate_insights(results, llm)

        # 4 · Build HTML report
        progress(0.88, desc="Building HTML report …")
        html_report = generate_html_report(results, insights, dataset_name)

        # 5 · Save report to disk for download
        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"{dataset_name}_eda_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_report)

        progress(1.0, desc="Done!")

        status = f"""✅  Analysis complete!

📂  Dataset  : {dataset_name}
📏  Shape    : {df.shape[0]:,} rows × {df.shape[1]} columns
🔢  Numeric  : {len(results['overview']['numeric_cols'])} columns
🏷  Categorical: {len(results['overview']['categorical_cols'])} columns
⚠  Columns with missing data : {sum(1 for v in results['overview']['missing_values'].values() if v > 0)}
🔁  Duplicate rows: {results['overview']['duplicate_rows']:,}

📄  Report saved — click Download below."""

        return html_report, status, report_path, gr.update(value="")

    except Exception as exc:
        import traceback
        return (
            None,
            f"❌  Error during analysis:\n{traceback.format_exc()}",
            None,
            gr.update(value=None),
        )


# ── Q&A function ──────────────────────────────────────────────────────────────
def answer_question(question: str, history: list):
    global _current_df, _df_context_str

    if not question.strip():
        return history

    if _current_df is None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "Please upload and analyse a CSV file first."})
        return history

    try:
        llm      = _get_llm()
        response = answer_data_question(question, _df_context_str, llm)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
    except Exception as exc:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"Error: {exc}"})

    return history


def clear_chat():
    return []


# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global overrides ── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
}

/* ── Hero header ── */
#hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    border-radius: 16px;
    padding: 36px 32px 28px;
    margin-bottom: 8px;
    border: 1px solid rgba(99, 102, 241, 0.25);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15);
}
#hero-header h1 {
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 40%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
    letter-spacing: -0.02em;
}
#hero-header p {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    line-height: 1.6;
}
#hero-header strong {
    color: #c7d2fe !important;
}

/* ── Upload area ── */
.upload-btn { min-height: 100px !important; }

/* ── Run button glow ── */
#run-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s ease !important;
}
#run-btn:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Status box ── */
#status-box textarea {
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
}

/* ── Section dividers ── */
#qa-divider {
    border-top: 1px solid rgba(99, 102, 241, 0.2);
    padding-top: 24px;
    margin-top: 12px;
}
#qa-divider h3 {
    color: #818cf8 !important;
    font-weight: 600 !important;
}

/* ── Report panel – force light background ── */
#report-panel .prose {
    background: #ffffff !important;
    color: #1e293b !important;
    border-radius: 12px;
    padding: 0 !important;
}
#report-panel iframe,
#report-panel .html-container {
    background: #ffffff !important;
    border-radius: 12px;
}

/* ── Chat styling ── */
#chatbot {
    border-radius: 12px !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
}

/* ── Ask button ── */
#ask-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    font-weight: 600 !important;
}

/* ── Footer ── */
#app-footer p {
    color: #64748b !important;
    font-size: 0.78rem !important;
}
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Data Analysis Copilot",
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🤖 Data Analysis Copilot
        **AI-powered Exploratory Data Analysis** — upload any CSV to get instant
        visualisations, statistics, and a downloadable HTML report written by
        **Groq LLaMA 3.3 70B** (completely free).
        """,
        elem_id="hero-header",
    )

    # ── Row 1: Upload + controls ──────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            file_input = gr.File(
                label="📂  Upload CSV",
                file_types=[".csv"],
                elem_classes=["upload-btn"],
            )
            target_input = gr.Textbox(
                label="🎯  Target column (optional)",
                placeholder="e.g.  Survived  or  price",
            )
            run_btn = gr.Button(
                "🚀  Run Full Analysis",
                variant="primary",
                size="lg",
                elem_id="run-btn",
            )
            status_box = gr.Textbox(
                label="Status",
                lines=10,
                interactive=False,
                buttons=["copy"],
                elem_id="status-box",
            )
            report_dl = gr.File(label="⬇  Download HTML Report", visible=True)

        with gr.Column(scale=3):
            report_out = gr.HTML(
                label="Analysis Report",
                value=(
                    "<div style='padding:60px 40px;text-align:center;"
                    "color:#94a3b8;font-size:1.1rem;'>"
                    "<p style='font-size:2.5rem;margin-bottom:16px;'>📊</p>"
                    "<p style='font-weight:500;'>Upload a CSV and click "
                    "<strong>Run Full Analysis</strong> to see the report here.</p>"
                    "</div>"
                ),
                elem_id="report-panel",
            )

    run_btn.click(
        fn=run_analysis,
        inputs=[file_input, target_input],
        outputs=[report_out, status_box, report_dl, target_input],
    )

    # ── Row 2: Chat Q&A ───────────────────────────────────────────────────────
    gr.Markdown(
        "### 💬 Ask Questions About Your Data",
        elem_id="qa-divider",
    )
    gr.Markdown(
        "After running analysis, ask anything: *'Which column has the most outliers?'*, "
        "*'What is the correlation between age and fare?'*, *'Suggest a good ML target column.'*"
    )

    chatbot = gr.Chatbot(
        height=340,
        label="Data Q&A",
        buttons=["copy"],
        elem_id="chatbot",
    )
    with gr.Row():
        q_input = gr.Textbox(
            placeholder="Ask anything about the dataset …",
            scale=5,
            show_label=False,
        )
        ask_btn   = gr.Button("Ask ↗", scale=1, elem_id="ask-btn")
        clear_btn = gr.Button("Clear", scale=1, variant="secondary")

    ask_btn.click(answer_question, inputs=[q_input, chatbot], outputs=[chatbot])
    q_input.submit(answer_question, inputs=[q_input, chatbot], outputs=[chatbot])
    clear_btn.click(clear_chat, outputs=[chatbot])

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(
        "<center>"
        "Data Analysis Copilot · Built with Python, LangChain, Groq, Matplotlib, Seaborn & Gradio"
        "</center>",
        elem_id="app-footer",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,           # set True to get a public Gradio link
        inbrowser=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
        allowed_paths=["/tmp", tempfile.gettempdir(), os.path.join(os.getcwd(), "reports")],
    )
