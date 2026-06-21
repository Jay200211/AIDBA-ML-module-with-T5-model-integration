"""Prompt templates for the LLM."""


OPTIMIZE_PROMPT = """You are an expert database performance engineer.
Given a slow query and its EXPLAIN plan from a {dialect} database, propose optimizations.

Return STRICT JSON with this exact schema:
{{
  "issues": [{{"type": str, "explanation": str, "severity": "low|medium|high"}}],
  "rewritten_query": "the full optimized SQL",
  "indexes_to_create": [{{"table": str, "columns": [str], "reason": str}}],
  "indexes_to_drop": [{{"table": str, "index": str, "reason": str}}],
  "estimated_impact_pct": int,
  "confidence": "low|medium|high",
  "notes": str
}}

Original query:
```sql
{query}
"""