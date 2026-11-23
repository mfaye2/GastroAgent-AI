
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gastroagent.multimodal import search_multimodal


def main():

    question = input("Describe a dish: ").strip()

    if not question:
        question = "A pizza with mozzarella and basil"

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

    results = search_multimodal(
        question=question,
        image_path=image_path,
        top_k=5
    )

    print("\n=== FUSION MULTIMODALE ===")

    for position, result in enumerate(results, start=1):

        print(f"\nRésultat {position}")
        print("Recette :", result["name"])
        print("Score :", round(result["score"], 6))
        print("Rang MiniLM :", result["text_rank"])
        print("Rang CLIP :", result["image_rank"])
        print("Image :", result["image_path"])


if __name__ == "__main__":
    main()
