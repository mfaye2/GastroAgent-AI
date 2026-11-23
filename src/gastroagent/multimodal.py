
from pathlib import Path

import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer

from gastroagent.config import CHROMA_DIR
from gastroagent.data_loader import load_augmented_recipes


def search_multimodal(
    question: str,
    image_path: Path | None = None,
    top_k: int = 5
):
    """Combine les classements MiniLM et CLIP."""

    if not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    # 1. Charger les deux modèles
    minilm = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    clip = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32"
    )

    # 2. Ouvrir les collections ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    recipes_db = client.get_collection(
        name="recipes",
        embedding_function=None
    )

    images_db = client.get_collection(
        name="food_images",
        embedding_function=None
    )

    # 3. Embedding textuel avec MiniLM
    text_vector = minilm.encode(
        question,
        normalize_embeddings=True
    )

    # 4. Embedding CLIP : image ou texte
    if image_path is not None:
        with Image.open(image_path) as img:
            clip_input = img.convert("RGB")
    else:
        clip_input = question

    image_vector = clip.encode(
        clip_input,
        normalize_embeddings=True
    )

    # 5. Recherche dans les deux collections
    text_results = recipes_db.query(
        query_embeddings=[text_vector.tolist()],
        n_results=min(10, recipes_db.count()),
        include=["metadatas", "distances"]
    )

    image_results = images_db.query(
        query_embeddings=[image_vector.tolist()],
        n_results=min(10, images_db.count()),
        include=["metadatas", "distances"]
    )

    # 6. Noms des recettes
    recipes = load_augmented_recipes()

    recipe_names = {
        int(recipe["id"]): recipe["name"]
        for recipe in recipes
    }

    # 7. Fusion par identifiant de recette
    scores = {}
    text_ranks = {}
    image_ranks = {}
    image_paths = {}

    # Résultats MiniLM
    for rank, metadata in enumerate(
        text_results["metadatas"][0], start=1
    ):
        recipe_id = int(metadata["recipe_id"])

        scores[recipe_id] = (
            scores.get(recipe_id, 0)
            + 0.6 / (60 + rank)
        )

        text_ranks[recipe_id] = rank

    # Résultats CLIP
    for rank, metadata in enumerate(
        image_results["metadatas"][0], start=1
    ):
        recipe_id = int(metadata["recipe_id"])

        scores[recipe_id] = (
            scores.get(recipe_id, 0)
            + 0.4 / (60 + rank)
        )

        image_ranks[recipe_id] = rank
        image_paths[recipe_id] = metadata["image_path"]

    # 8. Trier par score décroissant
    ranked_ids = sorted(
        scores,
        key=lambda recipe_id: scores[recipe_id],
        reverse=True
    )

    return [
        {
            "recipe_id": recipe_id,
            "name": recipe_names.get(recipe_id, "Unknown"),
            "score": scores[recipe_id],
            "text_rank": text_ranks.get(recipe_id),
            "image_rank": image_ranks.get(recipe_id),
            "image_path": image_paths.get(recipe_id)
        }
        for recipe_id in ranked_ids[:top_k]
    ]
