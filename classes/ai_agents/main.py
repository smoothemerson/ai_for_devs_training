# %%
from crewai import Agent, Crew, Task
from crewai.llm import LLM

# %%
llm = LLM(model="ollama/llama3.2:1b", base_url="http://localhost:11434")

# %%
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content about the {topic}",
    backstory="""
    You are working on planning a blog post article about the topic: {topic}.
    You collect information that helps the audience learn something and make informed decisions.
    Your work is the basis for the Content Writer to write the article on this topic.
    """,
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

# %%
writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory="""
    You are working on writing an opinion about the topic: {topic}.
    You base your opinion on the work of the Content Planner, who provides an outline and relevant content about the topic.
    You follow the main objectives and direction of the outline, as provided by the Content Planner.
    You also provide objective and impartial insights and back them with information provided by the Content Planner.    """,
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
# %%
editor = Agent(
    role="Editor",
    goal="Edit a given blog post article to align with the writing style of the organization.",
    backstory="""
    You are a editor that receives a blog post from the Content Writer.
    Your goal is to review the blog post to ensure that it follows journalistic best practices.
    """,
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
# %%
