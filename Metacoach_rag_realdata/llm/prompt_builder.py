# llm/prompt_builder.py

SYSTEM_PROMPT = """
You are MetaCoach, an elite cricket biomechanics coach.

STRICT RULES:
- Use ONLY the provided context.
- Never invent sessions, numbers, or metrics.
- If data is missing, say clearly that more data is required.
- Do not generalize beyond retrieved evidence.

Response format:
1) Quick Summary
2) What the metric means
3) Session-specific diagnosis (mean vs ideal_target)
4) What to fix + why it matters
5) Drills / prescription
6) Next session target
7) Comparison summary (if multi-session or multi-metric)
"""

def build_prompt(query, rag_results):

    context_blocks = []

    for r in rag_results:
        block = f"""
[Chunk]
Session: {r['metadata'].get('session_id')}
Metric: {r['metadata'].get('metric')}
Phase: {r['metadata'].get('phase')}
Text:
{r['text']}
"""
        context_blocks.append(block.strip())

    context = "\n\n".join(context_blocks)

    user_prompt = f"""
User Query:
{query}

Context:
{context}
"""

    full_prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_prompt}\n"
        f"<|assistant|>\n"
    )

    return full_prompt
