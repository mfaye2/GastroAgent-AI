
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chromadb
from sentence_transformers import SentenceTransformer

from gastroagent.config import CHROMA_DIR
from gastroagent.data_loader import load_augmented_recipes
from gastroagent.documents import build_recipe_documents


def main():

    # 1. Charger les recettes
    recipes = load_augmented_recipes()

    # 2. Construire les documents LangChain
    documents = build_recipe_documents(recipes)

    # 3. Charger MiniLM
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Calculer les embeddings
    texts = [doc.page_content for doc in documents]

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # 5. Créer le client ChromaDB local
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # 6. Créer ou récupérer la collection
    collection = client.get_or_create_collection(
        name="recipes",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}}
    )

    # 7. Enregistrer les documents et vecteurs
    collection.upsert(
        ids=[doc.id for doc in documents],
        documents=texts,
        metadatas=[doc.metadata for doc in documents],
        embeddings=embeddings.tolist()
    )

    print("\n=== CHROMADB ===")
    print("Collection :", collection.name)
    print("Documents stockés :", collection.count())
    print("Dossier :", CHROMA_DIR)


if __name__ == "__main__":
    main()
