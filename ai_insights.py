"""
AI Insights Generator
Uses Groq's free LLaMA 3.3 70B to analyse EDA results and write
a professional narrative report section.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a senior data scientist writing a professional EDA report.
You are precise, insightful, and data-driven. You always refer to actual numbers.
You write in clear paragraphs — not excessive bullet points.
You flag issues that would affect downstream ML modelling."""


def _trim(obj, max_chars: int = 3000) -> str:
    s = json.dumps(obj, default=str)
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


def generate_insights(eda_results: dict, llm: ChatGroq) -> str:
    """
    Feed EDA results to the LLM and get back a structured narrative.
    Returns a markdown-formatted string.
    """
    ov = eda_results.get("overview", {})

    prompt = f"""
You have just run a full EDA on a dataset. Here is the structured output:

---
DATASET SHAPE
Rows: {ov.get('n_rows', '?')} | Columns: {ov.get('n_cols', '?')}
Numeric columns  : {ov.get('numeric_cols', [])}
Categorical cols : {ov.get('categorical_cols', [])}
Memory usage     : {ov.get('memory_usage_mb', '?')} MB
Duplicate rows   : {ov.get('duplicate_rows', 0)}

MISSING VALUES (columns with >0% missing)
{_trim({k: f"{v} ({ov.get('missing_pct',{}).get(k,0)}%)"
        for k,v in ov.get('missing_values',{}).items() if v > 0})}

DESCRIPTIVE STATISTICS
{_trim(ov.get('describe', {}), 2500)}

TOP CORRELATIONS (feature pairs)
{_trim(eda_results.get('top_correlations', []))}

OUTLIER COUNTS (IQR method)
{_trim(eda_results.get('outlier_info', {}))}

DISTRIBUTION STATISTICS (skewness, kurtosis)
{_trim(eda_results.get('distribution_stats', {}))}
---

Write a professional EDA report with these exact sections.
Use **bold** for section titles, bullet points where helpful.

**1. Executive Summary**
2-3 sentences: what this dataset likely represents, overall data health.

**2. Key Statistical Findings**
5-8 specific findings using actual numbers from the data.
Mention notable distributions, ranges, dominant categories.

**3. Data Quality Assessment**
- Missing values: which columns, how severe, recommended treatment (drop/impute).
- Duplicates: note if significant.
- Outliers: which columns are most affected, recommended action.
- Data type issues: flag anything suspicious.

**4. Correlation Analysis**
Explain the top 3-5 most significant correlations found.
Note any potential multicollinearity issues for ML.

**5. Skewness & Distribution Notes**
Flag highly skewed features (|skew| > 1) and recommend transformations.

**6. Recommended Next Steps**
What ML tasks suit this data? What preprocessing pipeline would you recommend?
(e.g., encoding strategy, scaling, imputation, feature engineering ideas)

Be specific. Use numbers. Write like a professional."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content


def answer_data_question(question: str, df_context: str, llm: ChatGroq) -> str:
    """
    Answer a free-form question about the dataset.
    df_context is a string summary of the dataframe.
    """
    messages = [
        SystemMessage(content=f"""You are a helpful data scientist assistant.
The user has uploaded a CSV dataset. Here is the context:

{df_context}

Answer the user's question accurately and concisely using this data.
If a calculation is needed, do it step by step."""),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content
