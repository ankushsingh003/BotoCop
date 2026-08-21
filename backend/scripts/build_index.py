import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build-index")

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from backend.src.config import settings
from backend.src.rag.store import build_index, get_client_and_collection

def main():
    logger.info("Initializing ChromaDB index build...")
    
    data_dir = Path(settings.RAG_DATA_DIR)
    pdf_paths = [str(p) for p in data_dir.glob("*.pdf")]
    
    if not pdf_paths:
        logger.error(f"No PDF files found in data directory: {data_dir}")
        sys.exit(1)
        
    logger.info(f"Found {len(pdf_paths)} PDF(s) to index: {pdf_paths}")
    
    try:
        _, collection = get_client_and_collection()
        initial_count = collection.count()
        logger.info(f"Current document count in collection: {initial_count}")
        
        if initial_count > 0:
            logger.info("ChromaDB index already populated. Skipping indexing.")
            return
            
        build_index(pdf_paths)
        
        final_count = collection.count()
        logger.info(f"Index build completed. New document count in collection: {final_count}")
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
