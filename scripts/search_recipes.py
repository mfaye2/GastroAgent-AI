
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chromadb
from sentence_transformers import SentenceTransformer

from gastroagent.config import CHROMA_DIR


def main():

    # 1. Ouvrir ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name="recipes",
        embedding_function=None
    )

    # 2. Charger le même modèle MiniLM
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Question de l'utilisateur
    question = "Italian dish with mozzarella and basil"

    # 4. Transformer la question en vecteur
    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # 5. Chercher les 3 documents les plus proches
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    # 6. Afficher les résultats
    print("\nQuestion :", question)

    for i, doc_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        content = results["documents"][0][i]

        print(f"\n=== RESULTAT {i + 1} ===")
        print("ID :", doc_id)
        print("Cuisine :", metadata["cuisine"])
        print("Distance :", round(distance, 4))
        print("Texte :", content[:250])


if __name__ == "__main__":
    main()
