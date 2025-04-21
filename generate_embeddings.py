### USAGE: python generate_embeddings.py dataset.json

import json
import torch
import sys
from sentence_transformers import SentenceTransformer

def main(input_file):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.to(device)

    with open(input_file, "r") as f:
        dataset = json.load(f)

    sentence1s = [entry["sentence1"] for entry in dataset]
    sentence2s = [entry["sentence2"] for entry in dataset]

    # Do not pass device=..., let model manage it
    emb1 = model.encode(sentence1s, convert_to_tensor=True, show_progress_bar=True)
    emb2 = model.encode(sentence2s, convert_to_tensor=True, show_progress_bar=True)

    out_file = input_file.replace(".json", "_embeddings.pt")

    torch.save({
        "entries": dataset,         
        "emb1": emb1.cpu(),
        "emb2": emb2.cpu()
    }, out_file)

    print(f"Saved precomputed embeddings to '{out_file}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python precompute_embeddings.py <datafile.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    main(input_path)