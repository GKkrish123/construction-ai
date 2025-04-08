import os
import logging

MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "construction-ai-rag"
PERSIST_DIRECTORY = "./chroma_db"
UPLOAD_DIR = "./uploaded_docs"

NEO4J_URL = os.environ.get("NEO4J_URL", "neo4j+s://6599919e.databases.neo4j.io")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "your_password_here")
INDEX_NAME = "document_index"
DEFAULT_QUESTION = """Please analyze the whole document and extract the following details from it:
- Item/Fixture Types (e.g., pipe, fittings, valves, pumps, ductwork) as 'type'
- Quantities as 'quantities'
- Model Numbers / Specification References as 'model_no'
- Page References as 'page_ref'
- Associated Dimensions (if any) as 'dimensions'
- Mounting Type (wall-hung, floor-mounted, etc.) as 'mount_type'

Please return the structured data in a valid list of objects JSON format.
"""

os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
