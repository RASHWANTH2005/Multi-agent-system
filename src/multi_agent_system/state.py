import operator
from typing import Annotated

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from multi_agent_system.config import Step


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