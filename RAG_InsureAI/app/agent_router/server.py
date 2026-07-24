"""
Real MCP server for agent routing — a genuine FastMCP instance with
spec-correct tools/resources, runnable standalone over stdio for testing
with a real MCP client (the MCP Inspector via `mcp dev agent_router/server.py`,
Claude Desktop, or any other MCP client):

    python -m agent_router.server

api.py does NOT import from this file. The production hot path in
api.py calls registry.py/core.py directly as plain async Python — today
every registered agent (Ava) is invoked in-process, so routing a call
through this server's own tool-dispatch machinery would be a pure,
pointless overhead (a protocol round-trip to talk to code running in the
same process). This module exists so the routing logic is available as a
*real* MCP surface for anything that actually needs to reach it as an
out-of-process client — a future out-of-process agent orchestrator, an
ops script, the Inspector for manual testing — without duplicating any
logic: select_agent/invoke_agent below are thin wrappers around the exact
same core.select_agent() / registry.get_agent(...).invoke() functions
api.py calls directly.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import core, registry
import agent_router.agents  # noqa: F401 — import side-effect registers every agent

mcp = FastMCP("insurehub-agent-router")


@mcp.tool()
async def select_agent(query: str) -> dict:
    """Classify which specialist agent (if any) should handle this
    insurance query. Returns {"agent": str|None, "confidence": float,
    "method": "embedding"|"llm"|"none"}. agent=None/method="none" means
    the general insurance assistant (Layla) should keep handling it."""
    decision = await core.select_agent(query)
    return {"agent": decision.agent_name, "confidence": decision.confidence, "method": decision.method}


@mcp.tool()
async def invoke_agent(agent_name: str, session_id: str, message: str) -> str:
    """Invoke a registered specialist agent's own chat handler and return
    its reply text. Raises ValueError for an unknown agent_name."""
    agent = registry.get_agent(agent_name)
    if agent is None or agent.invoke is None:
        raise ValueError(f"Unknown or non-invokable agent: {agent_name!r}")
    return await agent.invoke(session_id, message)


@mcp.resource("agents://registry")
def agents_registry() -> list[dict]:
    """Introspection resource — every registered agent's name, display
    name, description, and example-query count. Lets an external MCP
    client (or a future ops dashboard) see what's routable without
    guessing tool arguments."""
    return [
        {
            "name": a.name,
            "display_name": a.display_name,
            "description": a.description,
            "example_count": len(a.example_queries),
        }
        for a in registry.all_agents()
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
