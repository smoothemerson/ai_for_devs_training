# %%
from crewai import Agent, Crew, Task
from crewai.llm import LLM
from IPython.display import Markdown

# %%
llm = LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")

# %%
roteirista = Agent(
    role="Roteirista de Vídeo",
    goal="Criar um roteiro detalhado e envolvente para um vídeo de games no YouTube sobre o tema: {query}",
    backstory="""
    Você é um roteirista especialista em vídeos de games para YouTube, com domínio em storytelling e pesquisa.
    Seu objetivo é pesquisar e elaborar um roteiro completo, informativo e cativante, adequado ao público gamer.
    Utilize ferramentas de pesquisa para enriquecer o conteúdo e garantir informações precisas e atuais.
    """,
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

# %%
criador_thumbnail = Agent(
    role="Criador de Thumbnail",
    goal="Criar três opções de thumbnails chamativas e criativas baseadas no roteiro do vídeo.",
    backstory="""
    Você é um designer gráfico especializado em thumbnails para vídeos de games no YouTube.
    Seu objetivo é criar três opções de thumbnails visualmente atrativas, inspiradas no roteiro do vídeo, destacando elementos que chamem a atenção do público gamer.
    """,
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

# %%
revisor = Agent(
    role="Revisor",
    goal="Revisar o roteiro e compilar o material final, incluindo o roteiro revisado e as thumbnails escolhidas.",
    backstory="""
    Você é um revisor experiente em conteúdo para YouTube.
    Seu objetivo é revisar o roteiro, corrigir eventuais erros e montar a versão final do material, incluindo o roteiro revisado e a melhor opção de thumbnail.
    """,
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

# %%
tarefa_roteiro = Task(
    description="""
      1. Pesquise e elabore um roteiro detalhado para um vídeo de games no YouTube sobre o tema: {query}.
      2. Estruture o roteiro com introdução, desenvolvimento e conclusão.
      3. Utilize técnicas de storytelling e dados relevantes para engajar o público.
    """,
    expected_output="Um roteiro detalhado e envolvente para um vídeo de games no YouTube.",
    agent=roteirista,
)

tarefa_thumbnails = Task(
    description="""
      1. Com base no roteiro gerado, crie três opções de thumbnails chamativas e criativas.
      2. As thumbnails devem destacar elementos visuais do conteúdo do vídeo e ser adequadas ao público gamer.
      3. Descreva cada thumbnail de forma clara e visual.
    """,
    expected_output="Três descrições detalhadas de thumbnails para o vídeo.",
    agent=criador_thumbnail,
)

tarefa_revisao = Task(
    description="""
      1. Revise o roteiro do vídeo, corrigindo eventuais erros e aprimorando o texto.
      2. Analise as três opções de thumbnails e escolha a melhor.
      3. Monte a versão final do material, incluindo o roteiro revisado e a thumbnail escolhida.
    """,
    expected_output="Material final contendo o roteiro revisado e a descrição da thumbnail escolhida.",
    agent=revisor,
)

# %%
crew = Crew(
    agents=[roteirista, criador_thumbnail, revisor],
    tasks=[tarefa_roteiro, tarefa_thumbnails, tarefa_revisao],
    verbose=True,
)

# %%
result = crew.kickoff(inputs={"query": "Melhores jogos de 2020"})

# %%
Markdown(result.raw)
