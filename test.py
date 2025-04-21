# USAGE: python run_similarity.py contextual_contrastive_model.pt

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model import ContextualContrastiveModel  
import sys

def show_similarity(model, sentence1, sentence2, context, device):
    model.eval()
    with torch.no_grad():
        emb1 = model([sentence1], [context])
        emb2 = model([sentence2], [context])
        sim = F.cosine_similarity(emb1, emb2).item()
        print(f"\nSimilarity between:\n- \"{sentence1}\"\n- \"{sentence2}\"\n→ Under context: \"{context}\"\n→ Cosine Similarity: {sim:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_similarity.py <model_weights.pt>")
        sys.exit(1)

    weights_path = sys.argv[1]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = ContextualContrastiveModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded model weights from: {weights_path}")

    s1 = "the red parrot is on the house"
    s2 = "the green parrot is flying over the river"
    context1 = "where the animal is."
    context2 = "the type of animal."

    show_similarity(model, s1, s2, context1, device)
    show_similarity(model, s1, s2, context2, device)