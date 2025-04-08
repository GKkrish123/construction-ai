# several configs are available in the deepdoctection repo to override the default settings
config_overwrite = [
    # "USE_TABLE_SEGMENTATION=False","USE_OCR=False"
]

import itertools
import deepdoctection as dd
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from constants import logging


def deepdoctection_pdf_loader(doc_path: str) -> list:
    """
    Process a PDF using deepdoctection and return a list of document pages with extracted text chunks.

    Parameters:
        doc_path (str): The file path to the PDF.
        use_gpu (bool): Whether to enable GPU usage (if available).
        score_threshold (float): Confidence threshold for detections.

    Returns:
        List[dict]: A list of dictionary objects, each containing:
                    - 'page_content': A string containing concatenated text from all chunks in that page.
                    - 'metadata': A dictionary with page-level metadata, e.g., page number and number of chunks.
    """
    try:
        analyzer = dd.get_dd_analyzer(config_overwrite=config_overwrite)

        df = analyzer.analyze(path=doc_path, verbose=True)
        df.reset_state()

        pages = list(df)

        documents = []
        for idx, page in enumerate(pages, start=1):
            chunk_texts = []
            for chunk in page.chunks:
                text = getattr(chunk, "text", str(chunk))
                chunk_texts.append(text)
            combined_text = "\n".join(chunk_texts)

            # Create a document dictionary resembling the output of UnstructuredPDFLoader.load()
            doc_dict = {
                "page_content": combined_text,
                # will store the pagenumber in the metadata for page reference extraction
                "metadata": {"page": idx, "num_chunks": len(page.chunks)},
            }
            documents.append(doc_dict)
        return documents
    except Exception as e:
        logging.error(f"Error deep doc processing PDF: {e}")


# Example usage:
if __name__ == "__main__":
    pdf_path = "./test_construction.pdf"
    documents = deepdoctection_pdf_loader(pdf_path)
    logging.info("Total pages processed:", len(documents))
    if documents:
        logging.info("Example page content (first page):")
        logging.info(documents[0]["page_content"])
