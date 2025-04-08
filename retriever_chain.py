from constants import logging
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    try:
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an expert in extracting structured data from plumbing submittal documents. 
        Extract the following details from the provided context:
        - Item/Fixture Types (e.g., pipe, fittings, valves, pumps, ductwork)
        - Quantities
        - Model Numbers / Specification References
        - Page References
        - Associated Dimensions (if any)
        - Mounting Type (wall-hung, floor-mounted, etc.)

        Please return the structured data in a valid JSON format.""",
        )
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )
        logging.info("Retriever created.")
        return retriever
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        raise


def create_chain(retriever, llm):
    """Create the RAG pipeline."""
    try:
        template = """Answer the question using only the following context:
    {context}
    Question: {question}
    """
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logging.info("Chain created...")
        return chain
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        raise
