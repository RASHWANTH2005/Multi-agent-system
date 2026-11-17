


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
    "give atleast 2 findings for each sub-question. "
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



# def numbered(questions: list[str]) -> str:
#     """0-based, to match Finding.sub_question_index."""
#     return "\n".join(f"{i}. {q}" for i, q in enumerate(questions))


# def research_system(plan: Plan | None, critique: str | None) -> str:
#     parts = [RESEARCH_PROMPT]
#     if plan:
#         parts.append("Cover these sub-questions:\n" + numbered(plan.sub_questions))
#     if critique:
#         parts.append("A reviewer found these problems with your previous attempt; fix them:\n" + critique)
#     return "\n\n".join(parts)


# def extract_system(plan: Plan | None) -> str:
#     if not plan:
#         return EXTRACT_PROMPT
#     return EXTRACT_PROMPT + "\n\nSub-questions:\n" + numbered(plan.sub_questions)
