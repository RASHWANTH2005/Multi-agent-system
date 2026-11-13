#client.py
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVERS = {
    # "fetch": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-fetch"]},
    "search": {"transport": "stdio", "command": "uvx", "args": ["duckduckgo-mcp-server"]},
}

_client = MultiServerMCPClient(SERVERS)
_tools = None

async def get_tools():
    global _tools
    if _tools is None:
        _tools = await _client.get_tools()
    return _tools

if __name__ == "__main__":
    import asyncio

    async def main():
        print([t.name for t in await get_tools()])

    asyncio.run(main())