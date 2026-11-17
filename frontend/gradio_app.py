import gradio as gr
import httpx


API_URL = "http://127.0.0.1:8000/api/v1/research"


async def research(topic):

    if not topic.strip():
        return "Please enter a research topic."

    async with httpx.AsyncClient(timeout=600) as client:

        response = await client.post(
            API_URL,
            json={"topic": topic}
        )

        response.raise_for_status()

        data = response.json()

        return data["report"]


demo = gr.Interface(
    fn=research,
    inputs=gr.Textbox(
        label="Research Topic",
        placeholder="Enter a topic, e.g. System Design"
    ),
    outputs=gr.Markdown(label="Research Report"),
    title="AI Technology Researcher",
    description="An AI multi-agent system for researching technical topics."
)


if __name__ == "__main__":
    demo.launch()