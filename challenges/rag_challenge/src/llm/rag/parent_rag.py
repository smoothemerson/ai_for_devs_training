from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama


def get_parent_rag_response(
    retriever: ParentDocumentRetriever, TEMPLATE: str, question: str, llm: ChatOllama
):
    rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)
    setup_retrieval = RunnableParallel(
        {"question": RunnablePassthrough(), "context": retriever}
    )

    output_parser = StrOutputParser()

    parent_retrieval_chain = setup_retrieval | rag_prompt | llm | output_parser

    response = parent_retrieval_chain.invoke(question)

    return response
