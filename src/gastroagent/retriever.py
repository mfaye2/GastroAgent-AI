
import unicodedata

import chromadb
from sentence_transformers import SentenceTransformer

from gastroagent.config import CHROMA_DIR


def normalize_text(text: str) -> str:
    """Met le texte en minuscules et retire les accents."""
    text = text.lower()

    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def select_collections(question: str) -> list[str]:
    """Sélectionne les collections selon les mots de la question."""

    q = normalize_text(question)

    selected = []

    restaurant_words = [
        "restaurant", "resto", "pizzeria",
        "ou manger", "aller manger", "where to eat"
    ]

    recipe_words = [
        "recette", "cuisiner", "preparer",
        "ingredients", "recipe", "how to cook"
    ]

    review_words = [
        "avis", "commentaire", "opinion",
        "review", "temoignage"
    ]

    if any(word in q for word in restaurant_words):
        selected.append("restaurants")

    if any(word in q for word in recipe_words):
        selected.append("recipes")

    if any(word in q for word in review_words):
        selected.append("reviews")

    # Si aucune intention reconnue, chercher partout
    return selected or ["restaurants", "recipes", "reviews"]


def detect_food(question: str) -> list[str]:
    """Identifie quelques plats français et anglais."""

    q = normalize_text(question)

    food_terms = {
        "pizza": ["pizza", "pizzas"],
        "sushi": ["sushi", "sushis"],
        "burger": ["burger", "burgers"],
        "pasta": ["pasta", "pates", "spaghetti"],
        "ramen": ["ramen"],
        "tacos": ["taco", "tacos"],
        "couscous": ["couscous"],
    }

    for words in food_terms.values():
        if any(word in q for word in words):
            return words

    return []


def search_all_collections( question: str,top_k: int = 3,collections: list[str] | None = None):

    # 1. Détecter les collections et le plat recherché
    collection_names = (
    collections
    if collections is not None
    else select_collections(question)
    )
    food_words = detect_food(question)

    # 2. Ouvrir ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # 3. Charger MiniLM
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Vectoriser la question
    query_vector = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    all_results = {}

    # 5. Rechercher dans les collections sélectionnées
    for name in collection_names:

        collection = client.get_collection(
            name=name,
            embedding_function=None
        )

        count = collection.count()

        if count == 0:
            all_results[name] = []
            continue

        # Chercher davantage de candidats pour les restaurants
        candidate_count = (
            max(top_k, 30)
            if name == "restaurants" and food_words
            else top_k
        )

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(candidate_count, count),
            include=["documents", "metadatas", "distances"]
        )

        matches = []

        for doc_id, content, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Vérifier le plat demandé dans les restaurants
            if name == "restaurants" and food_words:
                normalized_content = normalize_text(content)

                if not any(
                    word in normalized_content
                    for word in food_words
                ):
                    continue

            matches.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "distance": distance
            })

        # Conserver uniquement les premiers résultats
        all_results[name] = matches[:top_k]

    return all_results
