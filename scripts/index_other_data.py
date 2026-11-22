
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chromadb
from sentence_transformers import SentenceTransformer

from gastroagent.config import CHROMA_DIR

from gastroagent.data_loader import (
    load_structured_restaurants,
    load_augmented_reviews,
)

from gastroagent.documents import (
    build_restaurant_documents,
    build_review_documents,
)


def main():

    # 1. Charger MiniLM une seule fois
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Ouvrir la base ChromaDB existante
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # 3. Préparer les deux jeux de données
    datasets = [
        (
            "restaurants",
            load_structured_restaurants(),
            build_restaurant_documents,
        ),
        (
            "reviews",
            load_augmented_reviews(),
            build_review_documents,
        ),
    ]

    # 4. Indexer chaque jeu de données
    for collection_name, records, build_function in datasets:

        documents = build_function(records)

        if not documents:
            print(f"Collection {collection_name} : aucun document")
            continue

        texts = [doc.page_content for doc in documents]

        embeddings = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

        collection.upsert(
            ids=[doc.id for doc in documents],
            documents=texts,
            metadatas=[doc.metadata for doc in documents],
            embeddings=embeddings.tolist(),
        )

        print(f"\nCollection : {collection_name}")
        print("Documents :", collection.count())
        print("Dimensions :", embeddings.shape)

    # 5. Bilan des trois collections
    print("\n=== BILAN CHROMADB ===")

    for name in ["recipes", "restaurants", "reviews"]:
        collection = client.get_collection(name=name)
        print(f"{name} : {collection.count()} documents")


if __name__ == "__main__":
    main()
