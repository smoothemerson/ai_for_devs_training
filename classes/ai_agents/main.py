# %%
from crewai import Agent, Crew, Task
from crewai.llm import LLM
from IPython.display import Markdown

# %%
llm = LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")

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
plan = Task(
    description="""
      1. Prioritize the lasts trends, key players, and noteworthy news on {topic}.
      2. Identify the target audience, considering their interests and pain points.
      3. Develop a detaild content outline including an introduction, key points and call to action.
      4. Include SEO keyword and relevant data or sources.
    """,
    expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords and resources.",
    agent=planner,
)

write = Task(
    description="""
      1. Use the content plan to craft a compelling blog post on topic: {topic}.
      2. Incorporate the SEO keywords naturally.
      3. Sections/Subtitles are properly named in engaging manner.
      4. Ensure the post is structured with an engaging introduction, insightful body and summarization conclusion.
    """,
    expected_output="A well-written blog post in Markdown format ready for publication, each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description="""
      Proofread the given blog post for grammatical errors and aligment with the brand's voice.
    """,
    expected_output="A well-written blog post in Markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
    agent=editor,
)

# %%
crew = Crew(agents=[planner, writer, editor], tasks=[plan, write, edit], verbose=True)

# %%
result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})

# %%
Markdown(result.raw)
