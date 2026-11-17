from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI


MODEL_NAME = "gemini-3.5-flash-lite"  # defined model 
def get_llm():
    return ChatGoogleGenerativeAI(model=MODEL_NAME)

Step = Literal["planner", "researcher", "critic", "writer"] #supervisor's allowed steps, used in the graph type hints


# --- graph limits -----------------------------------------------------------
MAX_STEPS = 20        # supervisor hops before it forces the writer
RECURSION_LIMIT = 40    # LangGraph hard backstop (every node execution counts)


# --- MCP servers --------------------------------------------------------------
MCP_SERVERS = {
    # "fetch": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-fetch"]},
    "search": {"transport": "stdio", "command": "uvx", "args": ["duckduckgo-mcp-server"]},
}
