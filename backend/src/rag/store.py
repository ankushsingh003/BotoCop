import os
import chromadb
from langchain_core.documents import Document
from backend.src.rag.embedder import get_embedder

from chromadb import EmbeddingFunction

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        return get_embedder().encode(input).tolist()

    @staticmethod
    def name() -> str:
        return "local_embedding_function"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "LocalEmbeddingFunction":
        return LocalEmbeddingFunction()

def get_client_and_collection(domain: str = "general"):
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../backend/data/chroma_db"))
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    emb_fn = LocalEmbeddingFunction()
    
    normalized = domain.lower().replace("&", "and").replace(" ", "_")
    if normalized == "general":
        collection_name = "regulatory_rules"
    else:
        collection_name = f"rules_{normalized}"
        
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn
    )
    return client, collection

def build_index(domain: str = "general"):
    client, collection = get_client_and_collection(domain=domain)
    if collection.count() > 0:
        return
    
    from backend.src.config import settings
    from pathlib import Path
    data_dir = Path(settings.RAG_DATA_DIR)
    
    normalized = domain.lower().replace("&", "and").replace(" ", "_")
    
    if normalized == "general":
        pdf_files = [
            str(p) for p in data_dir.glob("*.pdf")
            if not p.name.startswith("rules_")
        ]
    else:
        target_pdf = data_dir / f"rules_{normalized}.pdf"
        if target_pdf.exists():
            pdf_files = [str(target_pdf)]
        else:
            pdf_files = []
            
    if not pdf_files:
        return
        
    from backend.src.rag.parser import parse_pdf_file
    for pdf_path in pdf_files:
        docs = parse_pdf_file(pdf_path)
        if not docs:
            continue
        
        documents = []
        metadatas = []
        ids = []
        for doc in docs:
            documents.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(f"{doc.metadata['source_file']}_{doc.metadata['chunk_index']}")
        
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

def get_retriever(query, top_k=5, domain: str = "general"):
    client, collection = get_client_and_collection(domain=domain)
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    docs = []
    if not results or not results["documents"] or not results["documents"][0]:
        return docs
        
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        docs.append(Document(page_content=text, metadata=meta))
    return docs

if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "backend" / "data"
    pdf_files = [str(p) for p in data_dir.glob("*.pdf")]
    build_index(pdf_files)
    results = get_retriever("influencer warning and disclosure rules", top_k=2)
    for r in results:
        print(r.metadata)
        print(r.page_content[:150].replace('\n', ' ') + "...")
