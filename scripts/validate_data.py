
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError

from gastroagent.data_loader import (
    load_augmented_recipes,
    load_augmented_reviews,
    load_structured_restaurants,
)

from gastroagent.schemas import (
    RecipeSchema,
    ReviewSchema,
    RestaurantSchema,
)


def validate_dataset(name, records, schema):

    valid_records = []
    errors = []

    for index, record in enumerate(records, start=1):

        try:
            validated = schema.model_validate(record)
            valid_records.append(validated)

        except ValidationError as error:
            errors.append((index, error))

    print(f"\n=== {name} ===")
    print("Total :", len(records))
    print("Valides :", len(valid_records))
    print("Invalides :", len(errors))

    # Afficher au maximum trois erreurs
    for index, error in errors[:3]:
        print(f"\nErreur à la ligne {index} :")
        print(error)

    return valid_records


def main():

    recipes = load_augmented_recipes()
    reviews = load_augmented_reviews()
    restaurants = load_structured_restaurants()

    valid_recipes = validate_dataset(
        "RECETTES", recipes, RecipeSchema
    )

    valid_reviews = validate_dataset(
        "AVIS CLIENTS", reviews, ReviewSchema
    )

    valid_restaurants = validate_dataset(
        "RESTAURANTS", restaurants, RestaurantSchema
    )


if __name__ == "__main__":
    main()
