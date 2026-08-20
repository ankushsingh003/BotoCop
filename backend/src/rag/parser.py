import os
import logging
from pathlib import Path
from typing import List, Union
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("rag-parser")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def parse_pdf_file(
    file_path: Union[str, Path], 
    chunk_size: int = 600, 
    chunk_overlap: int = 100
) -> List[Document]:
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    logger.info(f"Parsing PDF file: {file_path}")
    
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        logger.error(f"Failed to read PDF file {file_path}: {e}")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    documents = []
    chunk_index = 0
    source_file = file_path.name
    
    for page_idx, page in enumerate(reader.pages):
        page_number = page_idx + 1
        try:
            text = page.extract_text()
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_number} of {source_file}: {e}")
            continue
            
        if not text or not text.strip():
            continue
            
        chunks = text_splitter.split_text(text)
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                }
            )
            documents.append(doc)
            chunk_index += 1
            
    logger.info(f"Finished parsing {source_file}. Created {len(documents)} chunks.")
    return documents

def parse_pdf_directory(
    directory_path: Union[str, Path], 
    chunk_size: int = 600, 
    chunk_overlap: int = 100
) -> List[Document]:
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return []
        
    logger.info(f"Scanning directory for PDFs: {directory_path}")
    all_documents = []
    pdf_files = sorted(directory_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return []
        
    for pdf_file in pdf_files:
        docs = parse_pdf_file(
            file_path=pdf_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        all_documents.extend(docs)
        
    logger.info(f"Completed parsing all PDFs. Total chunks: {len(all_documents)}")
    return all_documents

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "backend" / "data"
    print(f"Testing PDF parser on directory: {data_dir}")
    documents = parse_pdf_directory(data_dir)
    print(f"Parsed {len(documents)} document chunks in total.")
    if documents:
        print(documents[0].metadata)
        print(documents[0].page_content[:200].replace('\n', ' ') + "...")
