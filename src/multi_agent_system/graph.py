from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from multi_agent_system.config import get_llm
from multi_agent_system.state import (
    State,
    Plan,
    Findings,
    Route
)

from multi_agent_system.client import get_tools

from multi_agent_system.agents.supervisor import (
    create_supervisor_node
)

from multi_agent_system.agents.planner import (
    create_planner_node
)

from multi_agent_system.agents.researcher import (
    create_researcher_node,
    create_extract_findings_node
)

from multi_agent_system.agents.critic import critic_node

from multi_agent_system.agents.writer import (
    create_writer_node
)


async def build_graph():

    tools = await get_tools()

    llm = get_llm()

    # Create model configurations
    router = llm.with_structured_output(Route)

    planner_llm = llm.with_structured_output(Plan)

    llm_with_tools = llm.bind_tools(tools)

    extractor = llm.with_structured_output(Findings)

    writer_chain = llm | StrOutputParser()


    # Create nodes
    supervisor = create_supervisor_node(router)

    planner = create_planner_node(planner_llm)

    researcher = create_researcher_node(llm_with_tools)

    extract_findings = create_extract_findings_node(extractor)

    writer = create_writer_node(writer_chain)

    # Build graph
    builder = StateGraph(State)

    builder.add_node("supervisor", supervisor)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("extractor", extract_findings)
    builder.add_node("critic", critic_node)
    builder.add_node("writer", writer)

  
    builder.add_edge(START, "supervisor")

    builder.add_edge("planner", "supervisor")
    builder.add_edge("extractor", "supervisor")
    builder.add_edge("critic", "supervisor")

    builder.add_conditional_edges(
        "researcher",
        tools_condition,
        {
            "tools": "tools",
            END: "extractor"
        }
    )

    builder.add_edge("tools", "researcher")

    builder.add_edge("writer", END)


    return builder.compile()