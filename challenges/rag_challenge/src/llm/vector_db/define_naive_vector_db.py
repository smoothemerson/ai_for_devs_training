from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_naive_vector_db(embedding_model: OllamaEmbeddings):
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory="./naiveVectorDB",
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=20, length_function=len, add_start_index=True
    )

    existing_count = vector_store._collection.count()

    if existing_count > 0:
        print(
            f"Naive vector database already exists with {existing_count} documents. Skipping initialization."
        )
    else:
        print("Initializing naive vector database from documents...")

        pdf_link = Path("document") / "sertoes_livro_euclides"
        loader = PyPDFLoader(pdf_link, extract_images=False)
        pages = loader.load_and_split()

        chunks = text_splitter.split_documents(pages)

        vector_store.add_documents(documents=chunks)

        print(
            f"Naive vector database initialization complete. Processed {len(pages)} document pages."
        )

    retriever_naive = vector_store.as_retriever(
        search_kwargs={"k": 3},
    )

    retriever_rerank = vector_store.as_retriever(
        search_kwargs={"k": 10},
    )

    return retriever_naive, retriever_rerank
