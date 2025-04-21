#USAGE: python generate_embeddings.py dataset.json

import torch
from torch.utils.data import Dataset, DataLoader
import random
import sys
from torch.nn.functional import cosine_similarity
import json

import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from torch.nn.functional import cosine_similarity

class ContrastiveBatchDataset(Dataset):
    def __init__(self, embedding_file, batch_size=32, threshold=0.5, device=None):
        data = torch.load(embedding_file)
        self.entries = data["entries"]
        self.emb1 = data["emb1"]
        self.emb2 = data["emb2"]
        self.context_keys = [e.get("context_key", "") for e in self.entries]
        self.slots1 = [e.get("slots1", {}) for e in self.entries]
        self.slots2 = [e.get("slots2", {}) for e in self.entries]
        self.batch_size = batch_size
        self.threshold = threshold

        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.emb1 = self.emb1.to(self.device)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return None

    def get_clean_batch(self):
        indices = list(range(len(self.entries)))
        random.shuffle(indices)

        selected = []
        selected_embs = []
        selected_keys = []
        selected_values = []

        for idx in indices:
            if len(selected) == self.batch_size:
                break

            emb = self.emb1[idx]
            ctx_key = self.context_keys[idx]
            ctx_value = self.slots1[idx].get(ctx_key, "")

            # Fast skip: conflict on context
            skip = False
            for k, v in zip(selected_keys, selected_values):
                if k == ctx_key and v == ctx_value:
                    skip = True
                    break
            if skip:
                continue

            if selected_embs:
                sims = cosine_similarity(emb.unsqueeze(0), torch.stack(selected_embs)).squeeze(0)
                if torch.any(sims >= self.threshold):
                    continue

            selected.append(idx)
            selected_embs.append(emb)
            selected_keys.append(ctx_key)
            selected_values.append(ctx_value)

        return [self.entries[i] for i in selected]

class FixedBatchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_batches):
        self.dataset = dataset
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataloader.py <embedding_file.pt>")
        sys.exit(1)

    embedding_file = sys.argv[1]

    dataset = ContrastiveBatchDataset(
        embedding_file=embedding_file,
        batch_size=32,
        threshold=0.8
    )

    def contrastive_collate_fn(_):
        return dataset.get_clean_batch()
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=contrastive_collate_fn
    )

    for batch in dataloader:
        print(f"Sample batch (first 2 entries):\n")
        print(json.dumps(batch, indent=2))
        break