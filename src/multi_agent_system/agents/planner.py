from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from multi_agent_system.state import  State

from multi_agent_system.prompts import PLAN_PROMPT


def create_planner_node(planner_llm):

    async def planner(state: State, config: RunnableConfig):
            plan = await planner_llm.ainvoke(
                [SystemMessage(PLAN_PROMPT)]
                + state["messages"],
                config,
            )
            return {"plan": plan,"findings": None, "critique": None}  # reset findings and critique if re-planning

    return planner