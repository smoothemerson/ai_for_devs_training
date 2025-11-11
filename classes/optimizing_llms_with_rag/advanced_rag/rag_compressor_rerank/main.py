# %%
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# %%
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# %%
llm = ChatOllama(
    model="gpt-oss:20b",
)

embeddings_model = OllamaEmbeddings(model="nomic-embed-text:v1.5")

# %%
# Carregar o PDF
pdf_link = "./rag/projeto_lei_ia.pdf"
loader = PyPDFLoader(pdf_link, extract_images=False)

pages = loader.load_and_split()
len(pages)

# %%
# Separar em Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=20, length_function=len, add_start_index=True
)

chunks = text_splitter.split_documents(pages)

# %%
# Salvar os chunks no vector db
vectordb = Chroma(embedding_function=embeddings_model, persist_directory="./naiveDB")

# %%
# Carregar o DB
naive_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# %%
rerank = CohereRerank(model="rerank-v3.5", top_n=3, cohere_api_key=COHERE_API_KEY)

compressor_retriever = ContextualCompressionRetriever(
    base_compressor=rerank,
    base_retriever=naive_retriever,
)

# %%
TEMPLATE = """
  Você é um especialista em legislação e tecnologia. Responda a pergunta abaixo utilizando o contexto informado.
  Query:
  {question}

  Context:
  {context}
"""

rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)

# %%`
setup_retrieval = RunnableParallel(
    {"question": RunnablePassthrough(), "context": compressor_retriever}
)

output_parser = StrOutputParser()

compressor_retrieval_chain = setup_retrieval | rag_prompt | llm | output_parser

# %%
compressor_retrieval_chain.invoke(
    "Quais os principais pontos de risco do marco legal de IA?"
)
