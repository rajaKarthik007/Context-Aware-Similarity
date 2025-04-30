import torch
import json
import random
from tqdm import tqdm
from torch.nn.functional import cosine_similarity

def build_triplet_eval_set_with_embeddings(embedding_file, output_file, similarity_threshold=0.5, max_attempts=100):
    data = torch.load(embedding_file)
    entries = data["entries"]
    emb1 = data["emb1"]
    emb2 = data["emb2"]
    context_keys = [e.get("context_key", "") for e in entries]
    slots1 = [e.get("slots1", {}) for e in entries]

    num_samples = len(entries)
    triplets = []

    print("Creating triplets...")

    for i in tqdm(range(num_samples)):
        anchor = entries[i]["sentence1"]
        positive = entries[i]["sentence2"]
        context = entries[i]["context"]
        ctx_key = context_keys[i]
        ctx_value = slots1[i].get(ctx_key, "")
        anchor_emb = emb1[i]

        found = False
        attempts = 0

        while not found and attempts < max_attempts:
            j = random.randint(0, num_samples - 1)
            if j == i:
                attempts += 1
                continue

            # Ensure context mismatch
            if context_keys[j] == ctx_key and slots1[j].get(ctx_key, "") == ctx_value:
                attempts += 1
                continue

            # Ensure semantic dissimilarity
            sim = cosine_similarity(anchor_emb.unsqueeze(0), emb1[j].unsqueeze(0)).item()
            if sim >= similarity_threshold:
                attempts += 1
                continue

            # Found a valid negative
            negative = random.choice([entries[j]["sentence1"], entries[j]["sentence2"]])
            triplets.append({
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "context": context
            })
            found = True

    with open(output_file, "w") as f:
        json.dump(triplets, f, indent=2)

    print(f"{len(triplets)} triplets saved to '{output_file}'")
    return triplets

# Example usage:
# python generate_triplets.py dataset_embeddings.pt triplet_eval_dataset.json
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_triplets.py <embedding_file.pt> <output_triplets.json>")
        sys.exit(1)

    embedding_file = sys.argv[1]
    output_file = sys.argv[2]
    build_triplet_eval_set_with_embeddings(embedding_file, output_file)