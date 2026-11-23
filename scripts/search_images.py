
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chromadb
from sentence_transformers import SentenceTransformer

from gastroagent.config import CHROMA_DIR


def main():

    question = input("Describe a food image: ").strip()

    if not question:
        question = "A pizza with mozzarella and basil"

    # Charger le même modèle CLIP
    model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32"
    )

    # Encoder le texte avec CLIP
    vector = model.encode(
        question,
        normalize_embeddings=True
    )

    # Ouvrir la collection contenant les images
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name="food_images",
        embedding_function=None
    )

    # Rechercher les 3 images les plus proches
    results = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=3,
        include=["metadatas", "distances"]
    )

    print("\nQuestion :", question)

    for index, metadata in enumerate(
        results["metadatas"][0], start=1
    ):

        distance = results["distances"][0][index - 1]

        print(f"\n=== IMAGE {index} ===")
        print("Recette :", metadata["recipe_name"])
        print("Fichier :", metadata["image_path"])
        print("Distance :", round(distance, 4))


if __name__ == "__main__":
    main()
