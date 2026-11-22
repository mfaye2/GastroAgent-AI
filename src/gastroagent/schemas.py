
from pydantic import BaseModel, Field, ConfigDict
from pydantic import BaseModel, Field, ConfigDict, field_validator

class RecipeSchema(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True)

    # Identification de la recette
    id: int = Field(gt=0)
    name: str = Field(min_length=1)
    cuisine: str = Field(min_length=1)

    # Informations pratiques
    servings: int = Field(gt=0)
    prep_time: str = Field(min_length=1)
    cook_time: str = Field(min_length=1)
    total_time: str = Field(min_length=1)

    # Contenu de la recette
    ingredients: list[str] = Field(min_length=1)
    directions: list[str] = Field(min_length=1)

    # Description générée par le LLM Vision
    image_description: str = Field(min_length=1)


class ReviewSchema(BaseModel):
    reviewId: int = Field(gt=0)
    userId: str = Field(min_length=1)
    itemId: int = Field(gt=0)

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    date: str = Field(min_length=1)

    rating: float = Field(ge=0, le=5)
    language: str = Field(min_length=1)

    images: str | list[str]
    image_captions: list[str]


class RestaurantSchema(BaseModel):
    name: str = Field(min_length=1)
    location: str = Field(min_length=1)

    type: str = Field(min_length=1)
    food_style: str | None = None

    @field_validator("food_style", mode="before")
    @classmethod
    def normalize_food_style(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    rating: float | None = Field(
        default=None, ge=0, le=5
    )

    price_range: int | None = Field(
        default=None, ge=1, le=5
    )

    signatures: list[str] = Field(default_factory=list)
    vibe: str | None = None
    environment: str = Field(min_length=1)
    shortcomings: list[str] = Field(default_factory=list)

    itemId: int | None = Field(default=None, gt=0)
