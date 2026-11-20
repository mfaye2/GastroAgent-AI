import json
from pathlib import Path
from gastroagent.config import (
    RAW_RESTAURANTS_FILE,
    RAW_RECIPES_FILE,
    RAW_REVIEWS_FILE,
    STRUCTURED_RESTAURANTS_FILE,
    AUGMENTED_RECIPES_FILE,
    AUGMENTED_REVIEWS_FILE,
)

def load_json(file_path: Path):
    """Charge un fichier JSON et retourne son contenu."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_raw_restaurants():
    return load_json(RAW_RESTAURANTS_FILE)

def load_raw_recipes():
    return load_json(RAW_RECIPES_FILE)

def load_raw_reviews():
    return load_json(RAW_REVIEWS_FILE)

def load_structured_restaurants():
    return load_json(STRUCTURED_RESTAURANTS_FILE)

def load_augmented_recipes():
    return load_json(AUGMENTED_RECIPES_FILE)

def load_augmented_reviews():
    return load_json(AUGMENTED_REVIEWS_FILE)