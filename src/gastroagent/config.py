from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dossiers principaux
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"

# Fichiers raw
RAW_RESTAURANTS_FILE = RAW_DIR / "restaurants_checkpoint.json"
RAW_RECIPES_FILE = RAW_DIR / "Recipes.json"
RAW_REVIEWS_FILE = RAW_DIR / "Synthetic-User-Reviews.json"

# Fichiers processed
STRUCTURED_RESTAURANTS_FILE = PROCESSED_DIR / "structured_restaurant_data.json"
AUGMENTED_RECIPES_FILE = PROCESSED_DIR / "augmented_food_recipe.json"
AUGMENTED_REVIEWS_FILE = PROCESSED_DIR / "augmented_user_review.json"
# Dossier de stockage de la base vectorielle
CHROMA_DIR = PROJECT_ROOT / "chroma_db"