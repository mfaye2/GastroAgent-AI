
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer

from gastroagent.config import CHROMA_DIR, IMAGES_DIR
from gastroagent.data_loader import load_augmented_recipes


def main():

    # 1. Charger les recettes pour obtenir leurs noms
    recipes = load_augmented_recipes()

    recipes_by_id = {
        int(recipe["id"]): recipe
        for recipe in recipes
    }

    # 2. Localiser les images PNG
    image_paths = sorted(
        IMAGES_DIR.rglob("recipe*.png")
    )

    if not image_paths:
        raise FileNotFoundError(
            "Aucune image trouvée dans data/images"
        )

    # 3. Charger le modèle CLIP
    model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32"
    )

    # 4. Ouvrir ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="food_images",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}}
    )

    # 5. Traiter les images par petits lots
    batch_size = 8

    for start in range(0, len(image_paths), batch_size):

        batch_paths = image_paths[start:start + batch_size]

        images = []
        ids = []
        metadatas = []
        descriptions = []

        for path in batch_paths:

            recipe_id = int(
                path.stem.removeprefix("recipe")
            )

            recipe = recipes_by_id[recipe_id]

            # Ouvrir l'image en RGB
            with Image.open(path) as img:
                images.append(img.convert("RGB"))

            ids.append(f"image-{recipe_id}")

            metadatas.append({
                "recipe_id": recipe_id,
                "recipe_name": recipe["name"],
                "image_path": path.relative_to(ROOT).as_posix()
            })

            descriptions.append(
                f"Image of {recipe['name']}"
            )

        # 6. Calculer les embeddings CLIP
        embeddings = model.encode(
            images,
            batch_size=batch_size,
            normalize_embeddings=True
        )

        # 7. Enregistrer le lot dans ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=descriptions,
            metadatas=metadatas
        )

        print(
            f"Images traitées : "
            f"{min(start + batch_size, len(image_paths))}"
            f"/{len(image_paths)}"
        )

    print("\n=== INDEXATION CLIP ===")
    print("Collection :", collection.name)
    print("Images stockées :", collection.count())
    print("Dimension :", model.get_sentence_embedding_dimension())


if __name__ == "__main__":
    main()
