from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from multi_agent_system.config import MAX_STEPS, Step
from multi_agent_system.prompts import SUPERVISOR_PROMPT
from multi_agent_system.state import State


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

def create_supervisor_node(llm):

    async def supervisor(state: State, config: RunnableConfig) -> Command[Step]:
            steps = state.get("steps", 0)
            allowed = allowed_steps(state)
    
            if steps >= MAX_STEPS:
                choice, reason = "writer", "step budget exhausted; writing with known gaps"
            elif len(allowed) == 1:
                choice, reason = allowed[0], "only valid next step"
            else:
                route = await llm.ainvoke(
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
    
    return supervisor
