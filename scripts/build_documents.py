
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gastroagent.data_loader import load_augmented_recipes
from gastroagent.documents import build_recipe_documents


def main():

    # Charger les 109 recettes JSON
    recipes = load_augmented_recipes()

    # Convertir les recettes en documents LangChain
    documents = build_recipe_documents(recipes)

    print("Nombre de documents :", len(documents))

    # Examiner le premier document
    first = documents[0]

    print("\n=== PAGE CONTENT ===")
    print(first.page_content[:600])

    print("\n=== METADATA ===")
    print(first.metadata)

    print("\n=== DOCUMENT ID ===")
    print(first.id)


if __name__ == "__main__":
    main()
