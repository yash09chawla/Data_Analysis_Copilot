"""
Report Generator
Compiles EDA results + AI insights into a polished, self-contained HTML file
that can be opened in any browser or shared with stakeholders.
"""

import re
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────────────────────
def _img(b64: str | None, alt: str = "chart") -> str:
    if not b64:
        return '<p class="no-data">No data available for this chart.</p>'
    return (f'<img src="data:image/png;base64,{b64}" '
            f'alt="{alt}" style="max-width:100%;border-radius:10px;'
            f'box-shadow:0 2px 12px rgba(0,0,0,0.06);">')


def _md_to_html(text: str) -> str:
    """Minimal markdown → HTML for the AI insights block."""
    # bold
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    # numbered section headings  e.g.  **1. Executive Summary**
    text = re.sub(
        r"^<strong>(\d+\.\s+[^<]+)<\/strong>$",
        r'<h4 class="insight-heading">\1</h4>',
        text, flags=re.MULTILINE,
    )
    # bullet lines
    lines, output, in_ul = text.split("\n"), [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_ul:
                output.append("<ul>")
                in_ul = True
            output.append(f"  <li>{stripped[2:]}</li>")
        else:
            if in_ul:
                output.append("</ul>")
                in_ul = False
            if stripped:
                output.append(f"<p>{stripped}</p>")
    if in_ul:
        output.append("</ul>")
    return "\n".join(output)


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Force a fully self-contained light theme ── */
:root {
  --bg-primary:    #f8fafc;
  --bg-card:       #ffffff;
  --bg-accent:     #f1f5f9;
  --bg-insight:    #fafbff;
  --text-primary:  #0f172a;
  --text-secondary:#475569;
  --text-muted:    #94a3b8;
  --border-light:  #e2e8f0;
  --border-accent: #c7d2fe;
  --brand-primary: #4f46e5;
  --brand-dark:    #312e81;
  --brand-light:   #eef2ff;
  --success:       #059669;
  --warning:       #d97706;
  --danger:        #dc2626;
  --radius:        12px;
  --shadow-sm:     0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-md:     0 4px 6px rgba(0,0,0,0.04), 0 2px 4px rgba(0,0,0,0.06);
}

body, html {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--brand-primary); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── Header ── */
.report-header {
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
  color: white;
  padding: 48px 40px 40px;
  text-align: center;
}
.report-header h1 {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 10px;
}
.report-header .sub {
  color: rgba(255,255,255,0.7);
  font-size: 0.9rem;
  font-weight: 400;
}
.report-header .sub strong {
  color: rgba(255,255,255,0.92);
}

/* ── Container ── */
.container {
  max-width: 1140px;
  margin: 0 auto;
  padding: 32px 24px 60px;
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

/* ── KPI strip ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 14px;
  margin: 28px 0;
}
.kpi {
  background: var(--bg-card);
  border-radius: var(--radius);
  padding: 22px 14px;
  text-align: center;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-light);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
.kpi .val {
  font-size: 2rem;
  font-weight: 700;
  color: var(--brand-primary);
  line-height: 1.2;
}
.kpi .lbl {
  font-size: 0.72rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 6px;
  font-weight: 500;
}

/* ── Section card ── */
.section {
  background: var(--bg-card) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius);
  padding: 30px 32px;
  margin: 20px 0;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-light);
}
.section h2 {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--brand-dark) !important;
  padding-bottom: 12px;
  margin-bottom: 20px;
  border-bottom: 2px solid var(--border-accent);
}
.section p, .section li, .section td, .section th, .section span {
  color: var(--text-primary) !important;
}

/* ── Insights ── */
.insights-body {
  background: var(--bg-insight) !important;
  border-left: 4px solid var(--brand-primary);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 24px 28px;
  line-height: 1.85;
  color: var(--text-primary) !important;
}
.insights-body h4.insight-heading {
  color: var(--brand-dark) !important;
  font-size: 1.05rem;
  font-weight: 600;
  margin: 22px 0 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border-light);
}
.insights-body h4.insight-heading:first-child {
  border-top: none;
  margin-top: 0;
  padding-top: 0;
}
.insights-body p  { margin: 8px 0; color: var(--text-primary) !important; }
.insights-body strong { color: var(--text-primary) !important; font-weight: 600; }
.insights-body ul { margin: 6px 0 12px 24px; }
.insights-body li { margin-bottom: 6px; color: var(--text-secondary) !important; }

/* ── Tables ── */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
  background: var(--bg-card) !important;
}
th {
  background: var(--brand-light) !important;
  color: var(--brand-dark) !important;
  padding: 12px 16px;
  text-align: left;
  font-weight: 600;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
td {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-light);
  color: var(--text-primary) !important;
}
tr:hover td { background: var(--bg-accent) !important; }
.green  { color: var(--success) !important; font-weight: 600; }
.amber  { color: var(--warning) !important; font-weight: 600; }
.red    { color: var(--danger)  !important; font-weight: 600; }
.no-data {
  color: var(--text-muted) !important;
  text-align: center;
  padding: 20px;
  font-style: italic;
}

/* ── Corr badges ── */
.corr-strong   { color: var(--danger)  !important; font-weight: 700; }
.corr-moderate { color: var(--warning) !important; font-weight: 700; }
.corr-weak     { color: var(--success) !important; font-weight: 600; }

/* ── Footer ── */
.footer {
  text-align: center;
  color: var(--text-muted) !important;
  font-size: 0.78rem;
  padding: 28px;
  border-top: 1px solid var(--border-light);
  margin-top: 8px;
}
"""


# ── Main builder ─────────────────────────────────────────────────────────────
def generate_html_report(
    eda_results: dict,
    ai_insights: str,
    dataset_name: str = "Dataset",
) -> str:

    ov       = eda_results.get("overview", {})
    missing  = ov.get("missing_values", {})
    miss_pct = ov.get("missing_pct", {})
    top_corr = eda_results.get("top_correlations", [])
    out_info = eda_results.get("outlier_info", {})
    now      = datetime.now().strftime("%d %b %Y, %H:%M")

    # ── Missing values table rows ────────────────────────────────────────────
    missing_rows = ""
    has_missing = any(v > 0 for v in missing.values())
    if has_missing:
        for col, cnt in sorted(missing.items(), key=lambda x: -x[1]):
            if cnt == 0:
                continue
            pct  = miss_pct.get(col, 0)
            cls  = "red" if pct > 20 else "amber" if pct > 5 else "green"
            rec  = ("Drop column" if pct > 50
                    else "Mean/median impute" if col in ov.get("numeric_cols", [])
                    else "Mode impute / 'Unknown'")
            missing_rows += (
                f"<tr><td>{col}</td>"
                f"<td>{cnt:,}</td>"
                f"<td class='{cls}'>{pct:.1f}%</td>"
                f"<td>{rec}</td></tr>"
            )
    else:
        missing_rows = '<tr><td colspan="4" class="green" style="text-align:center">✓ No missing values found</td></tr>'

    # ── Correlation table rows ────────────────────────────────────────────────
    corr_rows = ""
    for c1, c2, val in top_corr[:8]:
        cls  = "corr-strong" if abs(val) >= 0.7 else "corr-moderate" if abs(val) >= 0.4 else "corr-weak"
        note = ("Strong positive" if val >= 0.7
                else "Strong negative" if val <= -0.7
                else "Moderate positive" if val >= 0.4
                else "Moderate negative" if val <= -0.4
                else "Weak")
        corr_rows += (
            f"<tr><td>{c1}</td><td>{c2}</td>"
            f"<td class='{cls}'>{val}</td>"
            f"<td>{note}</td></tr>"
        )
    if not corr_rows:
        corr_rows = '<tr><td colspan="4" class="no-data">Not enough numeric columns for correlation.</td></tr>'

    # ── Outlier table rows ────────────────────────────────────────────────────
    out_rows = ""
    for col, info in out_info.items():
        pct = info.get("pct", 0)
        cls = "red" if pct > 10 else "amber" if pct > 3 else "green"
        rec = ("Cap / Winsorise" if pct > 5 else "Flag & monitor" if pct > 1 else "Likely fine")
        out_rows += (
            f"<tr><td>{col}</td>"
            f"<td>{info.get('count', 0):,}</td>"
            f"<td class='{cls}'>{pct:.1f}%</td>"
            f"<td>{rec}</td></tr>"
        )
    if not out_rows:
        out_rows = '<tr><td colspan="4" class="no-data">No outlier data available.</td></tr>'

    # ── Target section (optional) ─────────────────────────────────────────────
    target_section = ""
    if eda_results.get("target_img"):
        target_section = f"""
        <div class="section">
          <h2>🎯 Target Column Analysis</h2>
          {_img(eda_results['target_img'], 'target analysis')}
        </div>"""

    # ── Build HTML ────────────────────────────────────────────────────────────
    insights_html = _md_to_html(ai_insights)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Report — {dataset_name}</title>
<style>{CSS}</style>
</head>
<body style="background:#f8fafc !important; color:#0f172a !important;">

<div class="report-header">
  <h1>Exploratory Data Analysis Report</h1>
  <p class="sub">
    <strong>{dataset_name}</strong> &nbsp;·&nbsp;
    Generated: {now} &nbsp;·&nbsp;
    Powered by <strong>Groq LLaMA 3.3 70B</strong>
  </p>
</div>

<div class="container" style="background:#f8fafc !important; color:#0f172a !important;">

  <!-- KPI strip -->
  <div class="kpi-grid">
    <div class="kpi"><div class="val">{ov.get('n_rows',0):,}</div><div class="lbl">Rows</div></div>
    <div class="kpi"><div class="val">{ov.get('n_cols',0)}</div><div class="lbl">Columns</div></div>
    <div class="kpi"><div class="val">{len(ov.get('numeric_cols',[]))}</div><div class="lbl">Numeric</div></div>
    <div class="kpi"><div class="val">{len(ov.get('categorical_cols',[]))}</div><div class="lbl">Categorical</div></div>
    <div class="kpi"><div class="val">{ov.get('duplicate_rows',0):,}</div><div class="lbl">Duplicates</div></div>
    <div class="kpi"><div class="val">{sum(1 for v in missing.values() if v>0)}</div><div class="lbl">Cols w/ nulls</div></div>
    <div class="kpi"><div class="val">{ov.get('memory_usage_mb',0)}</div><div class="lbl">MB Memory</div></div>
  </div>

  <!-- AI Insights -->
  <div class="section">
    <h2>🤖 AI-Generated Insights &amp; Recommendations</h2>
    <div class="insights-body">{insights_html}</div>
  </div>

  <!-- Missing Values -->
  <div class="section">
    <h2>🔍 Missing Values Analysis</h2>
    <table>
      <thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Recommended Treatment</th></tr></thead>
      <tbody>{missing_rows}</tbody>
    </table>
    <br>{_img(eda_results.get('missing_img'), 'missing values chart')}
  </div>

  <!-- Distributions -->
  <div class="section">
    <h2>📊 Numeric Feature Distributions</h2>
    {_img(eda_results.get('distributions_img'), 'distributions')}
  </div>

  <!-- Correlations -->
  <div class="section">
    <h2>🔗 Feature Correlation Analysis</h2>
    <table style="margin-bottom:20px">
      <thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th><th>Strength</th></tr></thead>
      <tbody>{corr_rows}</tbody>
    </table>
    {_img(eda_results.get('correlations_img'), 'correlation heatmap')}
  </div>

  <!-- Outliers -->
  <div class="section">
    <h2>⚠ Outlier Analysis (IQR Method)</h2>
    <table style="margin-bottom:20px">
      <thead><tr><th>Column</th><th>Outlier Count</th><th>Outlier %</th><th>Recommendation</th></tr></thead>
      <tbody>{out_rows}</tbody>
    </table>
    {_img(eda_results.get('outliers_img'), 'outlier boxplots')}
  </div>

  <!-- Categoricals -->
  <div class="section">
    <h2>🏷 Categorical Feature Distributions</h2>
    {_img(eda_results.get('categorical_img'), 'categorical distributions')
     if eda_results.get('categorical_img')
     else '<p class="no-data">No suitable categorical columns (all have >30 unique values or no object columns).</p>'}
  </div>

  {target_section}

</div>

<div class="footer">
  Data Analysis Copilot &nbsp;·&nbsp; Built with Python, Matplotlib, Seaborn &amp; Groq LLaMA 3.3
</div>

</body>
</html>"""
    return html
