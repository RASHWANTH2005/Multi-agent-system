"""Research loop nodes: research_llm (decides + calls tools) and extract_findings (structures the result).

The tool node itself is wired in graph.py (langgraph's ToolNode).
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from multi_agent_system.prompts import RESEARCH_PROMPT, EXTRACT_PROMPT
from multi_agent_system.state import Findings, State


def create_researcher_node(llm_with_tools):

    async def researcher(state: State, config: RunnableConfig):
        sub_qs = "\n".join(f"- {i}: {q}" for i, q in enumerate(state["plan"].sub_questions))
        critique = state.get("critique")
        prompt = f"""
            {RESEARCH_PROMPT}
            Cover these sub-questions:{sub_qs}
            Previous critique:
            {critique or "No previous critique."}
            Address all issues identified in the previous critique.
            """

        #added a human message to ovrride the ai message after a critque cause gemini will not support model prefilling
        msgs = [SystemMessage(prompt)] + state["messages"] + [HumanMessage("Retify any critique if there and provide new findings.")]       
        response = await llm_with_tools.ainvoke(msgs, config)
        return {"messages": [response]}

    return researcher


def create_extract_findings_node(llm):

    async def extract_findings(state: State, config: RunnableConfig):
        research_summary = state["messages"][-1].content
        msgs = [SystemMessage(EXTRACT_PROMPT)] + [HumanMessage(research_summary)]
        new_findings = await llm.ainvoke(msgs, config)
        
        previous = (state["findings"].findings if state.get("findings") else [])

        unique = {
            (f.sub_question_index, f.claim, f.source_url): f
            for f in previous + new_findings.findings
        }

        return {
            "findings": Findings(findings=list(unique.values())),
            "critique": None
        }

    return extract_findings
