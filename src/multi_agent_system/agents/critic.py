from multi_agent_system.checks import validate_findings


async def critic_node(state):

    issues = validate_findings(
        plan=state["plan"],
        findings=state["findings"],
        messages=state["messages"]
    )

    return {
        "critique": "\n".join(issues) or "passed",
        "iterations": state.get("iterations", 0) + 1
    }