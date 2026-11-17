import asyncio

from multi_agent_system.graph import build_graph

from multi_agent_system.tracing import langfuse_handler, flush_traces


async def main():

    graph = await build_graph()
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "tell me about system design"}],
         "iterations": 0,
         "steps": 0,
         "decisions": []},
        config = {"callbacks": [langfuse_handler],"recursion_limit": 50 }
    )

    # print(result["findings"])
    print(result["plan"])
    print(result["critique"])
    print("#" * 20)
    print(result["report"])

    flush_traces()


if __name__ == "__main__":
    asyncio.run(main())