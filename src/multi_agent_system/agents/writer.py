from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from multi_agent_system.prompts import WRITE_PROMPT
from multi_agent_system.state import State



def create_writer_node(writer_chain):

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

    return writer