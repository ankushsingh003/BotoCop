import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.src.config import settings
from backend.src.rag.embedder import get_embedder
from backend.src.rag.store import get_client_and_collection, build_index
from pathlib import Path

logger = logging.getLogger("rag-retriever")

def mmr(query_embedding, candidate_embeddings, candidates, lambda_param=0.5, top_k=5):
    selected = []
    remaining = list(range(len(candidates)))

    while len(selected) < top_k and remaining:
        relevance = [float(cosine_similarity([query_embedding], [candidate_embeddings[i]])[0][0]) 
                     for i in remaining]
        
        if not selected:
            best = remaining[np.argmax(relevance)]
        else:
            redundancy = [max(float(cosine_similarity([candidate_embeddings[i]], 
                                                      [candidate_embeddings[s]])[0][0]) 
                             for s in selected) 
                          for i in remaining]
            
            mmr_scores = [lambda_param * rel - (1 - lambda_param) * red 
                          for rel, red in zip(relevance, redundancy)]
            best = remaining[np.argmax(mmr_scores)]
        
        selected.append(best)
        remaining.remove(best)
    
    return [candidates[i] for i in selected]

def trim_to_sentence_boundaries(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    starts_mid_sentence = False
    for char in text:
        if char.isalpha():
            if char.islower():
                starts_mid_sentence = True
            break

    if starts_mid_sentence:
        first_boundary = -1
        for i, char in enumerate(text):
            if char in ".!?":
                first_boundary = i
                break
        if first_boundary != -1 and first_boundary < len(text) - 1:
            text = text[first_boundary + 1:].strip()

    ends_mid_sentence = text[-1] not in ".!?\"'”’" if text else False

    if ends_mid_sentence:
        last_boundary = -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?":
                last_boundary = i
                break
        if last_boundary != -1:
            text = text[:last_boundary + 1].strip()

    if len(text) < 15:
        return text + "." if text and text[-1] not in ".!?\"'”’" else text

    return text

def sanitize_text(text: str) -> str:
    import re
    replacements = {
        "\ufffd": " ",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "•": "*",
        "⊲": "-",
        "\u22b2": "-",
        "\u201d": '"',
        "\u201c": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2022": "*",
        "\u2013": "-",
        "\u2014": "-"
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    
    text = "".join(ch if (32 <= ord(ch) <= 126 or ch in "\n\r\t") else " " for ch in text)
    text = re.sub(r' +', ' ', text)
    return text

def retrieve(query: str, domain: str = "general") -> str:
    normalized = domain.lower().replace("&", "and").replace(" ", "_")
    client, collection = get_client_and_collection(domain=domain)
    
    if collection.count() == 0:
        logger.info(f"ChromaDB '{domain}' collection is empty. Building index from PDF files...")
        build_index(domain=domain)
        client, collection = get_client_and_collection(domain=domain)

    if collection.count() == 0:
        logger.warning(f"No documents available in the RAG collection for domain '{domain}'.")
        return "No compliance rules found."

    logger.info(f"Embedding query: '{query}'")
    query_embedding = get_embedder().encode([query])[0].tolist()

    logger.info("Fetching candidate pool (top-20) from ChromaDB...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        include=["documents", "metadatas", "embeddings"]
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return "No compliance rules found matching query."

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    embs = results["embeddings"][0]

    candidates = []
    candidate_embeddings = []
    seen_texts = set()
    for i in range(len(docs)):
        text = docs[i].strip()
        if text not in seen_texts:
            seen_texts.add(text)
            candidates.append({"document": docs[i], "metadata": metas[i]})
            candidate_embeddings.append(embs[i])
    
    lambda_param = getattr(settings, "RAG_MMR_LAMBDA", 0.5)
    logger.info(f"Running MMR selection (lambda={lambda_param}) over {len(candidates)} unique candidates...")
    selected_candidates = mmr(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        candidates=candidates,
        lambda_param=lambda_param,
        top_k=min(5, len(candidates))
    )

    formatted_rules = []
    for idx, cand in enumerate(selected_candidates, 1):
        doc_text = sanitize_text(trim_to_sentence_boundaries(cand["document"]))
        source = cand["metadata"].get("source_file", "Unknown Source")
        page = cand["metadata"].get("page_number", "N/A")
        
        formatted_chunk = (
            f"Guideline {idx} [Source: {source}, Page {page}]:\n"
            f'<content>\n'
            f'{doc_text}\n'
            f'</content>'
        )
        formatted_rules.append(formatted_chunk)
    return "\n\n".join(formatted_rules)

class RuleRetriever:
    def retrieve(self, query: str, domain: str = "general") -> str:
        return retrieve(query, domain=domain)

if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    test_query = "disclosures requirements for brand endorsement and social media influencers"
    import sys
    retrieved_content = retrieve(test_query)
    print("\n=== RETRIEVED COMPLIANCE RULES (MMR) ===")
    safe_text = retrieved_content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
    print(safe_text)
