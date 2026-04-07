"""Query rewriting using Bedrock Claude for better retrieval."""
from bedrock_llm import chat_completion

REWRITE_SYSTEM = "You are a query rewriting assistant."
REWRITE_PROMPT = """Given a user query, rewrite it to be more specific and effective for document retrieval.
Return ONLY the rewritten query, nothing else.
If the query is already clear and specific, return it unchanged.

User query: {query}
Rewritten query:"""


def rewrite_query(query: str) -> str:
    return chat_completion(
        system_prompt=REWRITE_SYSTEM,
        user_prompt=REWRITE_PROMPT.format(query=query),
        temperature=0.0,
        max_tokens=200,
    )
