from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.stores import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_parent_vector_db(embedding_model: OllamaEmbeddings):
    store = InMemoryStore()
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory="./parentVectorDB",
    )

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    existing_count = vector_store._collection.count()

    if existing_count > 0:
        print(
            f"Parent vector database already exists with {existing_count} documents. Skipping initialization."
        )
    else:
        print("Initializing parent vector database from documents...")

        pdf_link = Path("document") / "sertoes_livro_euclides"
        loader = PyPDFLoader(pdf_link, extract_images=False)
        pages = loader.load_and_split()

        parent_document_retriever.add_documents(pages, ids=None)

        print(
            f"Parent vector database initialization complete. Processed {len(pages)} document pages."
        )

    return parent_document_retriever
