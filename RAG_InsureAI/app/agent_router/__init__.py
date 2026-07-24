"""
agent_router — multi-agent routing: which agent (Layla or a specialist bot
like Ava) should handle a given user query.

NAMING COLLISION WARNING: this is unrelated to app/router.py, which selects
the LLM *backend/provider* (vLLM vs Groq vs OpenAI vs Anthropic) for a
single agent's own generation calls. "router" here means "which agent
handles this conversation", not "which model serves this one call".

Replaces the single hardcoded regex (_TRAVEL_INTENT_PATTERN, formerly in
api.py) that could only ever recognize one specialist (Ava) with a real,
extensible registry (registry.py) plus a two-stage classifier: fast
embedding-similarity matching against each agent's description + example
queries (embeddings.py), falling back to a constrained LLM call only for
genuinely ambiguous queries (llm_fallback.py). core.select_agent() is the
single entry point tying both stages together.

Exposed as a real MCP server (server.py, FastMCP) so it can be driven by
an external MCP client (the Inspector, Claude Desktop, a future
out-of-process agent orchestrator) independent of this app — but the
production hot path in api.py calls registry.py/core.py directly as plain
Python, not through the MCP wire protocol, since that's a same-process
call today and paying a self-referential protocol round-trip for it would
be pure overhead. See server.py's own docstring.

Adding a new agent (health, life, etc.) later means one new file under
agents/ defining an AgentDefinition + invoke() and one import line in
agents/__init__.py — nothing here, in core.py, or in api.py's dispatch
logic needs to change. This is the "Phase 5: bot registry" work a prior
commit (02baeaa) explicitly deferred until a second bot existed.
"""
