from multi_agent_system import state
from multi_agent_system.state import Plan, Findings

def validate_findings(plan: Plan,findings: Findings,messages: list) -> list[str]:

    issues = []

    tool_text = "\n".join(str(m.content)for m in messages if m.type == "tool")

    bad = [f.source_url for f in findings.findings if f.source_url.rstrip("/") not in tool_text]

    if bad:
        issues.append(f"These URLs never appeared in any tool output: {bad}")

    covered = {f.sub_question_index for f in findings.findings}

    for i, q in enumerate(plan.sub_questions):
        if i not in covered:
            issues.append(
                f"Sub-question {i} has no findings: {q}"
            )

    return issues