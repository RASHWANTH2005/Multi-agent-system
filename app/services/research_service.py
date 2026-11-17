from multi_agent_system.graph import build_graph
from multi_agent_system.tracing import langfuse_handler, flush_traces

async def run_research(topic: str):

    graph = await build_graph()

    result = await graph.ainvoke(
        {
            "messages": [
                {"role": "user", "content": topic}
            ],
            "iterations": 0,
            "steps": 0,
            "decisions": []
        },
        config={
            "callbacks": [langfuse_handler],
            "recursion_limit": 50
        }
    )

    report = result.get("report", "")

    flush_traces()

    return report