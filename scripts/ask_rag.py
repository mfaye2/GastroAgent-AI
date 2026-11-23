
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gastroagent.rag import answer_with_rag


def main():

    question = input("Ask GastroAgent-AI: ").strip()

    if not question:
        print("La question ne peut pas être vide.")
        return

    image_input = input(
        "Image path (Enter pour aucune image) : "
    ).strip().strip('"')

    image_path = None

    if image_input:

        image_path = Path(image_input)

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        if not image_path.is_file():
            print("Image introuvable :", image_path)
            return

    # Appeler le RAG complet
    answer, details = answer_with_rag(
        question=question,
        image_path=image_path
    )

    print("\n=== REPONSE GASTROAGENT-AI ===")
    print(answer)

    print("\n=== SOURCES UTILISEES ===")

    for source in details["sources"]:
        print(
            source["collection"],
            "|",
            source["id"],
            "|",
            source["name"]
        )


if __name__ == "__main__":
    main()
