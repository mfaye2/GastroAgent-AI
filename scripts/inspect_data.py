import sys
from pathlib import Path

# Permet d'importer le package src/gastroagent
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gastroagent.data_loader import (
    load_augmented_recipes,
    load_augmented_reviews,
)
from gastroagent.config import IMAGES_DIR

def inspect_list(name, data):
    print(f"--- {name} ---")
    print("Type :", type(data).__name__)
    print("Nombre d'éléments :", len(data))
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        print("Champs :", list(data[0].keys()))
    print()

def main():
    recipes = load_augmented_recipes()
    reviews = load_augmented_reviews()
    images = list(IMAGES_DIR.glob("*.png"))

    inspect_list("processed/augmented_food_recipe.json", recipes)
    inspect_list("processed/augmented_user_review.json", reviews)

    print("Nombre d'images :", len(images))

if __name__ == "__main__":
    main()