import os
from constants import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from deepdoc import deepdoctection_pdf_loader


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        # for unstructured pdf loader
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()

        # for deepdoctection based pdf loader
        # data = deepdoctection_pdf_loader(doc_path)
        logging.info(f"PDF {doc_path} loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks
