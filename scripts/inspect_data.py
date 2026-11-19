
import json
from pathlib import Path

# 1. Chemin principal du projet
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 2. Fichiers JSON à examiner
files = [
    "raw/restaurants_checkpoint.json",
    "raw/Recipes.json",
    "raw/Synthetic-User-Reviews.json",
    "processed/structured_restaurant_data.json",
    "processed/augmented_food_recipe.json",
    "processed/augmented_user_review.json",
]

# 3. Examiner chaque fichier
for filename in files:
    path = DATA / filename
    print(f"\n--- {filename} ---")

    if not path.exists():
        print("Fichier introuvable")
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print("Type :", type(data).__name__)
    print("Nombre d'éléments :", len(data))

    # Afficher les clés du premier enregistrement
    if isinstance(data, list) and data:
        first = data[0]

        if isinstance(first, str):
            first = json.loads(first)

        if isinstance(first, dict):
            print("Champs :", list(first.keys()))

# 4. Vérifier les images
images = list((DATA / "images").rglob("recipe*.png"))
print("\nNombre d'images :", len(images))
