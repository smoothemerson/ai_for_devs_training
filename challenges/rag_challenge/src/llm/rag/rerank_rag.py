from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import ChatOllama


def get_rerank_rag_response(
    retriever: VectorStoreRetriever,
    TEMPLATE: str,
    question: str,
    llm: ChatOllama,
    COHERE_API_KEY,
):
    rerank = CohereRerank(model="rerank-v3.5", top_n=3, cohere_api_key=COHERE_API_KEY)

    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=rerank,
        base_retriever=retriever,
    )

    rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)
    setup_retrieval = RunnableParallel(
        {"question": RunnablePassthrough(), "context": compressor_retriever}
    )

    output_parser = StrOutputParser()

    naive_retrieval_chain = setup_retrieval | rag_prompt | llm | output_parser

    response = naive_retrieval_chain.invoke(question)

    return response
