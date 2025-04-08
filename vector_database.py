import os
import shutil
import ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from constants import EMBEDDING_MODEL, VECTOR_STORE_NAME, PERSIST_DIRECTORY, logging
import streamlit as st


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    try:
        ollama.pull(EMBEDDING_MODEL)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_db = None
        # for chroma database
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()

        # for neo 4j database
        # vector_db = Neo4jVector.from_existing_index(
        #     embedding=embedding,
        #     url=NEO4J_URL,
        #     username=NEO4J_USERNAME,
        #     password=NEO4J_PASSWORD,
        #     index_name=INDEX_NAME,
        # )
        logging.info("Loaded existing vector database.")
        return vector_db
    except Exception as e:
        logging.error(f"Error loading vector database: {e}")
        # st.error("Error loading vector database. Please check the logs.")
        return None


def clear_database():
    """Clear the vector database."""
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            logging.warning("Vector database cleared. Please upload documents again.")
    except Exception as e:
        logging.error(f"Error clearing vector database: {e}")
        st.error("Error clearing vector database. Please check the logs.")
