
import sys
import traceback
from pathlib import Path

import gradio as gr

# Racine du projet
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gastroagent.rag import answer_with_rag


IMAGES_DIR = (ROOT / "data" / "images").resolve()


def build_gallery(sources):
    """Prépare les photos des recettes pour Gradio."""

    gallery = []
    seen = set()

    for source in sources:

        if source.get("collection") != "recipes":
            continue

        # Chemin sauvegardé dans les métadonnées
        image_path = source.get("image_path")

        # Si absent, retrouver l'image par son ID
        if not image_path:
            recipe_id = str(source.get("id", ""))
            recipe_id = recipe_id.removeprefix("recipe-")

            if recipe_id.isdigit():
                image_path = (
                    IMAGES_DIR / f"recipe{recipe_id}.png"
                )

        if not image_path:
            continue

        path = Path(image_path)

        if not path.is_absolute():
            path = ROOT / path

        path = path.resolve()

        # Ne servir que les images du catalogue
        if not path.is_relative_to(IMAGES_DIR):
            continue

        if not path.is_file() or path in seen:
            continue

        seen.add(path)

        gallery.append((
            str(path),
            source.get("name", "Recette")
        ))

    return gallery


def respond(question, uploaded_image, history):
    """Relie l'interface Gradio au RAG existant."""

    history = list(history or [])
    question = (question or "").strip()

    if not question and not uploaded_image:
        return history, [], {}, "", None

    # Autoriser une recherche uniquement par photo
    search_question = question

    if not search_question:
        search_question = (
            "Find recipes visually similar to this image "
            "and explain the matching dishes."
        )

    displayed_question = question or "Recherche par image"

    if uploaded_image:
        displayed_question += "\n\n📷 Photo jointe"

    history.append({
        "role": "user",
        "content": displayed_question
    })

    try:

        # Appel à notre RAG multi-collections
        answer, details = answer_with_rag(
            question=search_question,
            image_path=(
                Path(uploaded_image)
                if uploaded_image else None
            )
        )

        history.append({
            "role": "assistant",
            "content": answer
        })

        gallery = build_gallery(
            details.get("sources", [])
        )

        return history, gallery, details, "", None

    except Exception:

        traceback.print_exc()

        history.append({
            "role": "assistant",
            "content": (
                "Une erreur s'est produite. "
                "Consulte le terminal VS Code pour les détails."
            )
        })

        return history, [], {}, "", None


# Création de l'interface
with gr.Blocks(
    title="GastroRAG — Multimodal Culinary Assistant"
) as demo:

    gr.Markdown(
        """
        # 🍽️ GastroRAG
        ### Multimodal Culinary RAG Assistant

        Explore restaurants, recipes and customer reviews
        using natural language and food images.

        **Powered by MiniLM, CLIP, ChromaDB, LangChain
        and OpenAI.**

        *Educational dataset — not live restaurant data.*
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                label="GastroRAG Assistant",
                height=450
            )

            question = gr.Textbox(
                label="Your question",
                placeholder=(
                    "Find an Italian restaurant, "
                    "recommend a recipe, or analyze a food image..."
                ),
                lines=2
            )

            with gr.Row():
                submit = gr.Button(
                    "🔍 Search",
                    variant="primary"
                )

                clear = gr.Button(
                    "Clear conversation"
                )

        with gr.Column(scale=1):

            image_input = gr.Image(
                label="Upload a food image (optional)",
                type="filepath",
                sources=["upload", "webcam", "clipboard"],
                height=260
            )

            gallery = gr.Gallery(
                label="Recommended food images",
                columns=2,
                height=300
            )

    with gr.Accordion(
        "Retrieval details and sources",
        open=False
    ):
        details = gr.JSON(
            label="Sources used by the RAG"
        )

    gr.Markdown(
        """
        **Example questions**

        - Find a restaurant serving Italian food.
        - Give me a recipe with mozzarella and basil.
        - What do customers say about the restaurants?
        - Recommend a dish similar to my uploaded picture.
        """
    )

    # Lancer la recherche
    submit.click(
        fn=respond,
        inputs=[question, image_input, chatbot],
        outputs=[chatbot, gallery, details, question, image_input]
    )

    question.submit(
        fn=respond,
        inputs=[question, image_input, chatbot],
        outputs=[chatbot, gallery, details, question, image_input]
    )

    # Réinitialiser l'interface
    clear.click(
        fn=lambda: ([], [], {}, "", None),
        outputs=[chatbot, gallery, details, question, image_input]
    )


if __name__ == "__main__":

    demo.queue(default_concurrency_limit=1)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[str(IMAGES_DIR)]
    )
