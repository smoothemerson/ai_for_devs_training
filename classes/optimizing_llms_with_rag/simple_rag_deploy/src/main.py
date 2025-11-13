import json

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load dos modelos (Embeddings e LLM)
llm = ChatOllama(model="gpt-oss:20b", temperature=0)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text:v1.5")

# Carregar Vector DB - Chroma
vector_store = Chroma(
    embedding_function=embeddings_model,
)


def loadData():
    # Carregar o PDF
    pdf_link = "../rag/document.pdf"
    loader = PyPDFLoader(pdf_link, extract_images=False)
    pages = loader.load_and_split()

    # Separar em Chunks (Pedações de documento)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=20, length_function=len, add_start_index=True
    )
    chunks = text_splitter.split_documents(pages)

    # Salvar no Vector DB - Chroma
    vector_store.add_documents(documents=chunks)

    # Carregar Retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3},
    )

    return retriever


def getRelevantDocuments(question: str):
    retriever = loadData()
    docs = retriever.invoke(question)
    return docs


def ask(question, llm):
    TEMPLATE = """"
      Você é um especialista em legislação e tecnologia. Responda a pergunta abaixo utilizando o contexto informado.

      Contexto: {contexto}

      Pergunta: {question}
    """

    prompt = PromptTemplate(input_variables=["contexto", "question"], template=TEMPLATE)

    sequence = RunnableSequence(prompt | llm)
    context = getRelevantDocuments(question)

    response = sequence.invoke({"contexto": context, "question": question})
    return response


def lambda_handler(event, context):
    body = json.loads(event.get("body", {}))
    query = body.get("question")
    response = ask(query, llm).content

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {"message": "Tarefa Concluída com sucesso", "details": response}
        ),
    }
