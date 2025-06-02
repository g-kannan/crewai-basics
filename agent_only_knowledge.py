from crewai import Agent, Task, Crew,LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Agent-specific knowledge
agent_knowledge = StringKnowledgeSource(
    content="snowflake-specific information that only this agent needs"
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(
    model="gemini/gemini-2.5-flash-preview-04-17",
    temperature=0.7,
)

agent = Agent(
    role="Specialist",
    goal="Use specialized knowledge",
    backstory="Expert with specific knowledge",
    knowledge_sources=[agent_knowledge],
    embedder={  # Agent can have its own embedder
        "provider": "google",
        "config": {"model": "text-embedding-004","api_key": os.environ["GEMINI_API_KEY"]},   
    },
    llm=llm,
)

task = Task(
    description="Answer using your specialized knowledge",
    agent=agent,
    expected_output="Overview on the data platform"
)

# No crew knowledge needed
crew = Crew(agents=[agent], tasks=[task],verbose=True)
result = crew.kickoff()  # Works perfectly