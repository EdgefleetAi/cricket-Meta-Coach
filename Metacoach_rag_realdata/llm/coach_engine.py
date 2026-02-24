# llm/coach_engine.py

from unified_query import unified_query
from llm.prompt_builder import build_prompt
from llm.generator import qwen_generate


def coach_with_qwen(query):

    rag_results = unified_query(query)

    if not rag_results:
        return "No relevant biomechanical data found."

    prompt = build_prompt(query, rag_results)

    response = qwen_generate(prompt)

    return response
