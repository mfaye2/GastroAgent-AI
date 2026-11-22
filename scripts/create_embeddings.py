
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import SentenceTransformer

from gastroagent.data_loader import load_augmented_recipes
from gastroagent.documents import build_recipe_documents


def main():

    # 1. Charger les données
    recipes = load_augmented_recipes()

    # 2. Construire les documents LangChain
    documents = build_recipe_documents(recipes)

    # 3. Récupérer les textes
    texts = [doc.page_content for doc in documents]

    # 4. Charger le modèle MiniLM
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Transformer les textes en vecteurs
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # 6. Examiner les résultats
    print("\nNombre de documents :", len(documents))
    print("Dimensions des embeddings :", embeddings.shape)

    print("\nPremier document :", documents[0].id)
    print("Premières valeurs du vecteur :")
    print(embeddings[0][:10])


if __name__ == "__main__":
    main()
