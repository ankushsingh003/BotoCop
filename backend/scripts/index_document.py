import os
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.src.rag.parser import parse_pdf_directory

logger = logging.getLogger("document-indexer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def index_docs():
    load_dotenv(override=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "../../backend/data")
    output_file = os.path.join(data_folder, "indexed_rules.json")
    
    logger.info("=" * 60)
    logger.info("Local Document Indexing:")
    logger.info(f"Source Folder: {data_folder}")
    logger.info(f"Output File: {output_file}")
    logger.info("=" * 60)

    logger.info(f"Parsing and chunking PDFs in: {data_folder}")
    try:
        splits = parse_pdf_directory(data_folder, chunk_size=600, chunk_overlap=100)
    except Exception as e:
        logger.error(f"Failed to parse and chunk PDF documents: {e}")
        return

    if splits:
        logger.info(f"Saving {len(splits)} chunks locally to {output_file}")
        try:
            serialized_docs = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in splits
            ]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(serialized_docs, f, indent=2, ensure_ascii=False)
            logger.info("=" * 60)
            logger.info(f"Successfully indexed and saved {len(splits)} chunks locally!")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to save indexed documents: {e}")
    else:
        logger.warning("No documents were processed")

if __name__ == "__main__":
    index_docs()
