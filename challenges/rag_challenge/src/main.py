import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.llm.ollama_declarations import embeddings_model, llm
from src.llm.rag.naive_rag import get_naive_rag_response
from src.llm.rag.parent_rag import get_parent_rag_response
from src.llm.rag.rerank_rag import get_rerank_rag_response
from src.llm.vector_db.define_naive_vector_db import setup_naive_vector_db
from src.llm.vector_db.define_parent_vector_db import setup_parent_vector_db


class ChatRequest(BaseModel):
    prompt: str


load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if COHERE_API_KEY is None:
    raise ValueError("COHERE_API_KEY environment variable not set")

retriever_naive = None
retriever_parent = None
retriever_rerank = None


TEMPLATE = """
  # Contexto
  Você é um especialista em literatura brasileira. Seu objetivo é ajudar a responder perguntas sobre o livro "Os Sertões" de Euclides da Cunha.

  # Tarefa
  Analise a pergunta e utilize o contexto fornecido para formular uma resposta precisa e informativa.

  # Formato da Resposta
  Forneça uma resposta em um texto corrido. Não use quebras de linhas nem listas. Não use negritos nem itálicos.

  Pergunta:
  {question}

  Contexto:
  {context}
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever_naive
    global retriever_parent
    global retriever_rerank
    print("Initializing vector database...")
    retriever_naive, retriever_rerank = setup_naive_vector_db(embeddings_model)
    retriever_parent = setup_parent_vector_db(embeddings_model)

    print("Vector database initialized successfully!")

    yield


app = FastAPI(
    title="Desafio RAG - Os Sertões",
    description="Essa é uma API para interagir com o modelo RAG baseado no livro Os Sertões de Euclides da Cunha.",
    contact={"name": "Emerson Rocha", "email": "emersonfaria019@gmail.com"},
    openapi_url="/openapi.json",
    docs_url="/api/docs",
    lifespan=lifespan,
)


@app.get("/healthcheck", summary="Health Check", tags=["Health"])
def healthcheck():
    return {
        "status": "OK",
    }


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/api/docs")


@app.post("/chat/naive_rag")
def chat_naive(request: ChatRequest):
    try:
        if retriever_naive is None:
            return {"error": "Vector database not initialized"}

        question = request.prompt
        naive_rag_response = get_naive_rag_response(
            retriever_naive, TEMPLATE, question, llm
        )

        return {
            "response": naive_rag_response,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat/parent_rag")
def chat_parent(request: ChatRequest):
    try:
        if retriever_parent is None:
            return {"error": "Vector database not initialized"}

        question = request.prompt
        parent_rag_response = get_parent_rag_response(
            retriever_parent, TEMPLATE, question, llm
        )

        return {
            "response": parent_rag_response,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat/rerank_rag")
def chat_rerank(request: ChatRequest):
    try:
        if retriever_rerank is None:
            return {"error": "Vector database not initialized"}

        question = request.prompt
        rerank_rag_response = get_rerank_rag_response(
            retriever_rerank, TEMPLATE, question, llm, COHERE_API_KEY
        )

        return {
            "response": rerank_rag_response,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
