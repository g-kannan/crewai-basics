from crewai import LLM, Agent, Crew, Process, Task
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a knowledge source from web content
content_source = CrewDoclingSource(
    file_paths=[
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking",
        "https://docs.databricks.com/gcp/en/release-notes/runtime/16.4lts",
    ],
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(
    model="gemini/gemini-2.5-flash-preview-04-17",
    temperature=0.7,
)


# Create an agent with the knowledge store
agent = Agent(
    role="About papers",
    goal="You know everything about the papers.",
    backstory="You are a master at understanding papers and their content.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

task = Task(
    description="Answer the following questions about the papers: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[content_source],
    embedder={  # Agent can have its own embedder
        "provider": "google",
        "config": {"model": "text-embedding-004","api_key": os.environ["GEMINI_API_KEY"]},   
    },
    llm=llm,
)

result = crew.kickoff(
    inputs={"question": "When the latest DBR was released?."}
)