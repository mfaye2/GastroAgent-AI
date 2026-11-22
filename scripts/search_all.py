
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gastroagent.retriever import search_all_collections


def main():

    question = input("Pose ta question : ").strip()

    if not question:
        question = "Italian food with basil and cozy atmosphere"

    results = search_all_collections(
        question=question,
        top_k=3
    )

    print("\nQUESTION :", question)

    for collection_name, matches in results.items():

        print(f"\n=== {collection_name.upper()} ===")

        for index, match in enumerate(matches, start=1):

            print(f"\nRésultat {index}")
            print("ID :", match["id"])
            print("Distance :", round(match["distance"], 4))
            print("Metadata :", match["metadata"])
            print("Texte :", match["content"][:200])


if __name__ == "__main__":
    main()
