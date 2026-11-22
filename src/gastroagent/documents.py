
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



#on ajoute les deux autres fonctions 


def build_restaurant_documents(restaurants: list[dict]) -> list[Document]:
    """Convertit les restaurants en documents LangChain."""

    documents = []

    for index, data in enumerate(restaurants, start=1):

        restaurant = RestaurantSchema.model_validate(data)

        content = (
            f"Restaurant: {restaurant.name}\n"
            f"Location: {restaurant.location}\n"
            f"Food style: {restaurant.food_style or 'Not specified'}\n"
            f"Type: {restaurant.type}\n"
            f"Signature dishes: {', '.join(restaurant.signatures)}\n"
            f"Vibe: {restaurant.vibe or 'Not specified'}\n"
            f"Environment: {restaurant.environment}\n"
            f"Shortcomings: {', '.join(restaurant.shortcomings)}"
        )

        metadata = {
            "name": restaurant.name,
            "location": restaurant.location,
            "source": "structured_restaurant_data.json",
        }

        if restaurant.food_style is not None:
            metadata["food_style"] = restaurant.food_style

        if restaurant.rating is not None:
            metadata["rating"] = restaurant.rating

        if restaurant.price_range is not None:
            metadata["price_range"] = restaurant.price_range

        document = Document(
            id=f"restaurant-{index}",
            page_content=content,
            metadata=metadata,
        )

        documents.append(document)

    return documents


def build_review_documents(reviews: list[dict]) -> list[Document]:
    """Convertit les avis clients en documents LangChain."""

    documents = []

    for data in reviews:

        review = ReviewSchema.model_validate(data)

        content = (
            f"Review title: {review.title}\n"
            f"Review: {review.text}\n"
            f"Image captions: {', '.join(review.image_captions)}"
        )

        metadata = {
            "review_id": review.reviewId,
            "item_id": review.itemId,
            "rating": review.rating,
            "language": review.language,
            "source": "augmented_user_review.json",
        }

        document = Document(
            id=f"review-{review.reviewId}",
            page_content=content,
            metadata=metadata,
        )

        documents.append(document)

    return documents

