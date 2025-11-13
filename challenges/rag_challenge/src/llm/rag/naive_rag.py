from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import ChatOllama


def get_naive_rag_response(
    retriever: VectorStoreRetriever, TEMPLATE: str, question: str, llm: ChatOllama
):
    rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)
    setup_retrieval = RunnableParallel(
        {"question": RunnablePassthrough(), "context": retriever}
    )

    output_parser = StrOutputParser()

    naive_retrieval_chain = setup_retrieval | rag_prompt | llm | output_parser

    response = naive_retrieval_chain.invoke(question)

    return response
