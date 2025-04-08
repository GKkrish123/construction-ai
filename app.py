import os
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import ollama
import torch
import streamlit as st
from constants import (
    DEFAULT_QUESTION,
    EMBEDDING_MODEL,
    UPLOAD_DIR,
    PERSIST_DIRECTORY,
    MODEL_NAME,
    logging,
)
from document_processing import ingest_pdf, split_documents
from vector_database import load_vector_db
from retriever_chain import create_retriever, create_chain

# for torch issue
torch.classes.__path__ = []

# for model pull
# ollama.pull(MODEL_NAME)

def get_answer(user_input=DEFAULT_QUESTION):
    """Get answer from the model using the retriever and chain."""
    try:
        llm = ChatOllama(model=MODEL_NAME)
        vector_db = load_vector_db()

        if len(vector_db.get()["ids"]) == 0:
            st.error("No documents found! Please upload a PDF first.")
            return

        # for neo4j vector store
        # test_results = vector_db.similarity_search(user_input, k=1)
        # if not test_results:
        #     st.error("No results matching! Please upload a PDF first.")
        #     return

        retriever = create_retriever(vector_db, llm)
        chain = create_chain(retriever, llm)
        response = chain.invoke(input=user_input)

        st.markdown("**💡 Assistant:**")
        st.write(response)
        logging.info("Answer generated successfully.")
    except Exception as e:
        if "unable to open database" in str(e):
            st.error("Database not found! Please upload a PDF first.")
        else:
            st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in getting answer: {str(e)}")


def process_and_store_document(file_path):
    """Processes and stores the uploaded document into the vector database."""
    try:
        vector_db = load_vector_db()
        data = ingest_pdf(file_path)

        if data is None:
            st.error("Error processing the document.")
            return
        chunks = split_documents(data)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_db.add_documents(
            documents=chunks,
            embedding=embedding,
        )
        logging.info("New document added to the vector database.")
        st.success("Document processed and stored successfully!")
    except Exception as e:
        st.error(f"An error occurred while processing the document: {str(e)}")
        logging.error(f"Error in processing document: {str(e)}")

def main():
    try:
        st.set_page_config(
            page_title="Construction AI Document Assistant", layout="wide"
        )
        st.title("📄 AI-Powered Construction Supply Assistant")
        st.markdown(
            "Upload real-world construction PDF, get the fixture info and ask questions about their contents."
        )

        with st.sidebar:
            st.header("📂 Upload Construction PDF")
            uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

            if uploaded_file:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File {uploaded_file.name} uploaded successfully.")
                process_and_store_document(file_path)

            if st.button("Clear Database", help="Deletes all stored documents."):
                if os.path.exists(PERSIST_DIRECTORY):
                    shutil.rmtree(PERSIST_DIRECTORY)
                st.warning("Vector database cleared. Please upload documents again.")

        # for initial Fixtures Response for which this app is intended for
        with st.spinner("Generating Fixture Data..."):
            try:
                get_answer()
                retrieved_fixture_data = True
            except Exception as e:
                st.error(f"An error occurred while fetching fixture data: {str(e)}")

        st.subheader("🔍 Ask a different Question")
        user_input = st.text_input("Enter your question:")

        if user_input:
            with st.spinner("Generating response..."):
                try:
                    get_answer(user_input)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.info("Enter a question to start querying your documents.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    main()
