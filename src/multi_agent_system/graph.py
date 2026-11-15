#graph.py
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode, tools_condition

import operator
from typing import Annotated, Literal
from langgraph.types import Command
from multi_agent_system.client import get_tools

load_dotenv()


langfuse = Langfuse()
langfuse_handler = CallbackHandler()

# print(langfuse.auth_check())

Step = Literal["planner", "researcher", "critic", "writer"]
MAX_STEPS = 20

class Route(BaseModel):
    next: Step
    reason: str = Field(description="One sentence citing specific facts from the state")

class Plan(BaseModel):
    sub_questions: list[str] = Field(
        description="2 specific sub-questions that together answer the user's question"
    )

class Finding(BaseModel):
    sub_question_index: int = Field(description="Index of the sub-question this finding addresses")
    claim: str = Field(description="One detailed factual claim")
    source_url: str = Field(description="URL of the page that supports this claim")


class Findings(BaseModel):
    findings: list[Finding]


class State(MessagesState):
    plan : Plan | None
    findings: Findings | None
    report: str | None
    critique: str | None
    iterations: int = 0
    steps: int
    decisions: Annotated[list[dict], operator.add]

PLAN_PROMPT = (
    "Break the user's question into 2 specific, researchable sub-questions. "
)
RESEARCH_PROMPT = (
    "You are an AI technology researcher. Use the search and fetch tools to "
    "research the topic. Base everything on pages you actually saw."
    "When calling the search tool, always set max_results=3. "
    "mention the source URL for each claim you make "
    ""
)
EXTRACT_PROMPT = (
    "From the research above, extract findings as claim + source_url. "
    "For each finding, include the sub-question index it addresses. "
    "Use zero-based indexing: first sub-question = 0, second = 1. "
    "Only use URLs that appear in the conversation. "
    "Do not invent claims or source URLs."
)
WRITE_PROMPT = (
    "Write a concise report. Use ONLY the findings provided. "
    "2 detailed summaray is enough. "
    "Cite the URL after each summary. If the findings don't cover something, say so."
    "if no findings were collected, say so."
)   
SUPERVISOR_PROMPT = (
    "You route a research pipeline. Choose the next step from the allowed list.\n"
    "- researcher: the plan is fine but evidence is missing or weak.\n"
    "- planner: the sub-questions themselves are too vague or wrong.\n"
    "- writer: remaining gaps can't reasonably be fixed; the report will state them.\n"
    "Your reason must cite specific facts from the status."
)



def allowed_steps(state: State) -> list[str]:
    if not state.get("plan"):             return ["planner"]
    if not state.get("findings"):         return ["researcher"]
    if state.get("critique") is None:     return ["critic"]
    if state["critique"] == "passed":     return ["writer"]
    return ["researcher", "planner", "writer"]   # critic found problems: real choice

def status(state: State) -> str:
    plan, fnd = state.get("plan"), state.get("findings")
    n_src = len({f.source_url for f in fnd.findings}) if fnd else 0
    return (
        f"sub-questions: {len(plan.sub_questions) if plan else 0}\n"
        f"claims: {len(fnd.findings) if fnd else 0} from {n_src} distinct sources\n"
        f"critic says: {state.get('critique') or 'passed'}\n"
        f"steps used: {state.get('steps', 0)} of {MAX_STEPS}"
    )

async def build_graph():
    tools = await get_tools()
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    planner_llm = llm.with_structured_output(Plan)
    llm_with_tools = llm.bind_tools(tools)
    extractor = llm.with_structured_output(Findings)
    writer_chain = llm | StrOutputParser()
    router = llm.with_structured_output(Route)


    async def supervisor(state: State, config: RunnableConfig) -> Command[Step]:
        steps = state.get("steps", 0)
        allowed = allowed_steps(state)

        if steps >= MAX_STEPS:
            choice, reason = "writer", "step budget exhausted; writing with known gaps"
        elif len(allowed) == 1:
            choice, reason = allowed[0], "only valid next step"
        else:
            route = await router.ainvoke(
                [SystemMessage(SUPERVISOR_PROMPT),
                HumanMessage(f"Question: {state['messages'][0].content}\n\n"
                            f"Status: {status(state)}\n\nAllowed: {allowed}")],
                config,
            )
            if route.next in allowed:
                choice, reason = route.next, route.reason
            else:
                choice, reason = allowed[0], f"overrode invalid choice '{route.next}'"

        return Command(
            goto=choice,
            update={"steps": steps + 1,
                    "decisions": [{"step": choice, "reason": reason}]},
        )


    async def planner(state: State, config: RunnableConfig):
        plan = await planner_llm.ainvoke(
            [SystemMessage(PLAN_PROMPT)]
            + state["messages"],
            config,
        )
        return {"plan": plan,"findings": None, "critique": None}  # reset findings and critique if re-planning

    async def research_llm(state: State, config: RunnableConfig):
        sub_qs = "\n".join(f"- {q}" for q in state["plan"].sub_questions)
        critique = state.get("critique")
        prompt = f"""
            {RESEARCH_PROMPT}
            Cover these sub-questions:
            {sub_qs}
            Previous critique:
            {critique or "No previous critique."}
            Address all issues identified in the previous critique.
            """

        msgs = [SystemMessage(prompt)] + state["messages"] + [HumanMessage("Retify any critique if there and provide new findings.")]       
        response = await llm_with_tools.ainvoke(msgs, config)
        return {"messages": [response]}

    async def extract_findings(state: State, config: RunnableConfig):
        research_summary = state["messages"][-1].content
        msgs = [SystemMessage(EXTRACT_PROMPT)] + [HumanMessage(research_summary)]
        new_findings = await extractor.ainvoke(msgs, config)
        
        previous = (state["findings"].findings if state.get("findings") else [])

        unique = {
            (f.sub_question_index, f.claim, f.source_url): f
            for f in previous + new_findings.findings
        }

        return {
            "findings": Findings(findings=list(unique.values())),
            "critique": None
        }

    async def critic(state: State):
        issues = []
        tool_text = "\n".join(str(m.content) for m in state["messages"] if m.type == "tool")

        bad = [f.source_url for f in state["findings"].findings
            if f.source_url.rstrip("/") not in tool_text]
        if bad:
            issues.append(f"These URLs never appeared in any tool output: {bad}")

        covered = {f.sub_question_index for f in state["findings"].findings}
        for i, q in enumerate(state["plan"].sub_questions):
            if i not in covered:
                issues.append(f"Sub-question {i} has no findings: {q}")

        return {"critique": "\n".join(issues) or "passed",
                "iterations": state.get("iterations", 0) + 1}

    # def route_after_critic(state: State):
    #     if state["critique"] and state["iterations"] < 3: 
    #         return "researcher"
    #     return END

    async def writer(state: State, config: RunnableConfig):
        question = state["messages"][0].content
        findings = state.get("findings")

        evidence = "\n".join(
            f"- {f.claim} [{f.source_url}]"
            for f in (findings.findings if findings else [])
        )

        if not evidence:
            evidence = "No findings were collected."

        report = await writer_chain.ainvoke(
            [
                SystemMessage(WRITE_PROMPT),
                HumanMessage(f"Question: {question}\n\nFindings:\n{evidence}"),
            ],
            config,
        )
        return {"report": report}



    builder = StateGraph(State)

    builder.add_node("supervisor", supervisor)
    builder.add_node("planner", planner)
    builder.add_node("researcher", research_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("extractor", extract_findings)
    builder.add_node("critic", critic)
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
   
    langfuse.flush()
    


if __name__ == "__main__":
    asyncio.run(main())