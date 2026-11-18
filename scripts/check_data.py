
import json
from pathlib import Path

# Localiser le dossier racine du projet
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Charger les fichiers JSON
with open(DATA / "raw" / "restaurants_checkpoint.json", encoding="utf-8") as f:
    restaurants = json.load(f)

with open(DATA / "raw" / "Recipes.json", encoding="utf-8") as f:
    recipes = json.load(f)

with open(DATA / "raw" / "Synthetic-User-Reviews.json", encoding="utf-8") as f:
    reviews = json.load(f)

# Compter les images
images = list((DATA / "images").glob("*.png"))

# Afficher les informations
print("Restaurants :", len(restaurants))
print("Recettes :", len(recipes))
print("Avis :", len(reviews))
print("Images :", len(images))

print("\nPremière recette :", recipes[0]["name"])
