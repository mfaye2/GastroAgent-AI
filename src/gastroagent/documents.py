
from langchain_core.documents import Document

from gastroagent.schemas import (
    RecipeSchema,
    RestaurantSchema,
    ReviewSchema,
)


def build_recipe_documents(recipes: list[dict]) -> list[Document]:
    """Transforme les recettes JSON en documents LangChain."""

    documents = []

    for recipe_data in recipes:

        # 1. Valider la recette avec Pydantic
        recipe = RecipeSchema.model_validate(recipe_data)

        # 2. Préparer les ingrédients et instructions
        ingredients = ", ".join(recipe.ingredients)
        directions = "\n".join(recipe.directions)

        # 3. Construire le texte destiné aux embeddings
        content = (
        f"Recipe: {recipe.name}\n"
        f"Cuisine: {recipe.cuisine}\n"
        f"Image description: {recipe.image_description}\n"
        f"Ingredients: {ingredients}"
            )

        # 4. Informations pour identifier et filtrer le document
        metadata = {
            "recipe_id": recipe.id,
            "cuisine": recipe.cuisine,
            "source": "augmented_food_recipe.json",
            "image_file": f"recipe{recipe.id}.png",
        }

        # 5. Créer un document LangChain
        document = Document(
            id=f"recipe-{recipe.id}",
            page_content=content,
            metadata=metadata,
        )

        documents.append(document)

    return documents
