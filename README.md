## AI-Powered Construction Supply Assistant

### Overview

The AI-Powered Construction Supply Assistant is a Streamlit-based application designed to facilitate the ingestion, processing, and querying of PDF documents. Users can upload real-world construction PDFs like plumbing submittal PDFs, which are then processed and stored in a vector database, enabling efficient retrieval and interaction through natural language queries.

### System Design - https://whimsical.com/construction-ai-design-DxkjsWtHAC4tKmMoRWJRuo

### How It Works

1. **Document Upload**: Users upload PDF documents via the Streamlit interface. The application saves these files to a designated directory and used as document context.
2. **Document Ingestion**: The uploaded PDFs are processed using document loaders such as `deepdoctection_pdf_loader` or `UnstructuredPDFLoader`. This step extracts the textual content from the PDFs.
3. **Text Splitting**: The extracted text is divided into manageable chunks using the `RecursiveCharacterTextSplitter`, facilitating efficient storage and retrieval.
4. **Embedding Generation**: Each text chunk is converted into vector embeddings using the `OllamaEmbeddings` model, preparing them for storage in the vector database (nomic-embed-text model).
5. **Vector Database Storage**: The embeddings are stored in a (Neo4j or Chroma) vector database, allowing for quick similarity searches and retrieval.
6. **Query Processing**: Users input questions through the Streamlit interface. The application utilizes a language model (`ChatOllama`) and a multi-query retriever to fetch relevant document chunks from the vector database.
7. **Response Generation**: The retrieved information is processed by the language model to generate the necessary JSON format response and display a coherent response to the user's queries.

### Environment Setup

* Setting up the environment
  1. Visual Studio Code (`https://code.visualstudio.com/download`) - Choose you own IDE
  2. Python 3.11.1 (`https://www.python.org/downloads/`)
  5. Create virtual environment by running the command `python -m venv <env_name>` or any other virtual env of your choice.
  6. Run the command `pip install -r requirements.txt` to install all Python dependencies.

* In this project `deepdocdetection` is setup with PyTorch and Tesseract OCR. For setting up with different packages of `deepdocdetection`, check out - https://deepdoctection.readthedocs.io/en/latest/install/.

### Deploying the APP server
  * Run the command `streamlit run app.py` to start the app.
    * http://localhost:8504 -> Streamlit UI URL

### Technologies Used

- **Streamlit**: Provides the user interface for document upload and interaction.
- **UnstructuredPDFLoader**: Extracts text from PDF documents.
- **deepdoctection**: An alternative tool for document extraction and analysis.
- **RecursiveCharacterTextSplitter**: Splits large text into smaller chunks for processing.
- **OllamaEmbeddings**: Generates vector embeddings from text chunks.
- **Chroma**: Serves as the vector database for storing and retrieving embeddings.
- **ChatOllama**: A language model used for generating responses based on user queries.
- **MultiQueryRetriever**: Enhances retrieval by reformulating user queries to fetch relevant document chunks.

### Limitations and Assumptions

- **Document Format**: The application currently supports only PDF documents. Other formats are not processed.
- **Embedding Model**: The quality of responses depends on the performance of the `OllamaEmbeddings` model. Variations in model accuracy can affect retrieval effectiveness.
- **Assumption of Content Quality**: The system assumes that the extracted text from PDFs is accurate and complete. Poor-quality scans or OCR errors can impact performance.
- **Dependency on External Services**: The application relies on external services and models, such as `OllamaEmbeddings`, which may introduce latency or availability issues.

---
