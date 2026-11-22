
import chromadb
from sentence_transformers import SentenceTransformer

from gastroagent.config import CHROMA_DIR


def search_all_collections(question: str, top_k: int = 3):
    """Recherche dans les collections textuelles de ChromaDB."""

    # 1. Ouvrir la base vectorielle
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # 2. Charger le modèle d'embeddings
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Transformer la question en vecteur
    query_vector = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # 4. Collections à interroger
    collection_names = [
        "restaurants",
        "recipes",
        "reviews"
    ]

    all_results = {}

    # 5. Interroger chaque collection
    for name in collection_names:

        collection = client.get_collection(
            name=name,
            embedding_function=None
        )

        if collection.count() == 0:
            all_results[name] = []
            continue

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matches = []

        # 6. Construire une liste de résultats lisibles
        for doc_id, content, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            matches.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "distance": distance
            })

        all_results[name] = matches

    return all_results
