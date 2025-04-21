# ###USAGE: python model.py dataset_embeddings.pt

# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, AutoModel
# import torch.nn.functional as F
# from dataloader import ContrastiveBatchDataset, FixedBatchDataset
# from torch.utils.data import DataLoader, Subset
# import sys
# import random
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# class ContextualContrastiveModel(nn.Module):
#     def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2", proj_dim=128):
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
#         self.encoder = AutoModel.from_pretrained(encoder_name)
#         for param in self.encoder.parameters():
#             param.requires_grad = False

#         hidden_size = self.encoder.config.hidden_size

#         # MLP that transforms context embedding before adding to sentence
#         self.context_mapper = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
#         )

#         self.context_gate = nn.Linear(hidden_size, hidden_size)
#         # Projection head after context injection
#         self.projection = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, proj_dim)
#         )

#     def mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output.last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
#         return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#     def encode_text(self, texts):
#         # Tokenize and encode
#         encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.encoder.device)
#         output = self.encoder(**encoded)
#         return self.mean_pooling(output, encoded["attention_mask"])

#     def forward(self, sentences, contexts):
#         sent_emb = self.encode_text(sentences)
#         ctx_emb = self.encode_text(contexts)

#         # ctx_mod = self.context_mapper(ctx_emb)
#         # mod_emb = sent_emb + self.context_mapper(ctx_emb) + (sent_emb * self.context_mapper(ctx_emb))
#         gate = torch.sigmoid(self.context_gate(ctx_emb))
#         mod_emb = (1 - gate) * sent_emb + gate * self.context_mapper(ctx_emb)
#         mod_emb = F.normalize(mod_emb, dim=1)

#         return self.projection(mod_emb)

#     def encode_pair(self, sent1, ctx1, sent2, ctx2):
#         z1 = self.forward(sent1, ctx1)
#         z2 = self.forward(sent2, ctx2)
#         return z1, z2


# def contrastive_loss(anchor_emb, positive_emb, temperature=0.1):
#     anchor = F.normalize(anchor_emb, dim=1)
#     positive = F.normalize(positive_emb, dim=1)
#     logits = torch.matmul(anchor, positive.T) / temperature
#     labels = torch.arange(len(anchor)).to(anchor.device)
    
#     loss_a_to_p = F.cross_entropy(logits, labels)
#     loss_p_to_a = F.cross_entropy(logits.T, labels)
    
#     return (loss_a_to_p + loss_p_to_a) / 2

# def evaluate(model, dataloader):
#     model.eval()
#     losses = []
#     with torch.no_grad():
#         for batch in dataloader:
#             anchor_texts = [item["sentence1"] for item in batch]
#             positive_texts = [item["sentence2"] for item in batch]
#             contexts = [item["context"] for item in batch]
#             anchor_emb = model(anchor_texts, contexts)
#             positive_emb = model(positive_texts, contexts)
#             loss = contrastive_loss(anchor_emb, positive_emb)
#             losses.append(loss.item())
#     model.train()
#     return sum(losses) / len(losses)

# def show_similarity(model, sentence_a, sentence_b, context_str):
#     with torch.no_grad():
#         emb1 = model([sentence_a], [context_str])
#         emb2 = model([sentence_b], [context_str])
#         sim = F.cosine_similarity(emb1, emb2).item()
#     print(f"\nSimilarity between:\n- \"{sentence_a}\"\n- \"{sentence_b}\"\n→ Under context: \"{context_str}\"\n→ Cosine Similarity: {sim:.4f}")


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python train_contrastive.py <embedding_file.pt>")
#         sys.exit(1)

#     embedding_file = sys.argv[1]
#     batch_size = 256

#     # Dataset & dynamic batching
#     current_dataset = ContrastiveBatchDataset(embedding_file, batch_size=batch_size, threshold=0.8)

#     def contrastive_collate_fn(_):
#         return current_dataset.get_clean_batch()

#     # Train-val split
#     total_len = len(current_dataset.entries)
#     indices = list(range(total_len))
#     random.shuffle(indices)
#     split = int(0.75 * total_len)
#     train_indices = indices[:split]
#     val_indices = indices[split:]

#     train_dataset = Subset(current_dataset, train_indices)
#     val_dataset = Subset(current_dataset, val_indices)

#     train_batches = len(train_indices) // batch_size
#     val_batches = len(val_indices) // batch_size

#     # DataLoaders
#     train_loader = DataLoader(FixedBatchDataset(train_dataset, train_batches), batch_size=1, collate_fn=contrastive_collate_fn, shuffle=False)
#     val_loader = DataLoader(FixedBatchDataset(val_dataset, val_batches), batch_size=1, collate_fn=contrastive_collate_fn, shuffle=False)

#     # Model + optimizer
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     model = ContextualContrastiveModel().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
#      # 🔍 Test contextual similarity
#     sentence1 = "The dog is covered in black paint"
#     sentence2 = "The cat has black paint all over it"
#     show_similarity(model, sentence1, sentence2, "the color of the animal")
#     show_similarity(model, sentence1, sentence2, "the type of animal")

#     # Training loop
#     num_epochs = 300
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#         epoch_loss = 0.0
#         progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

#         for step, batch in enumerate(progress_bar):
#             anchor_texts = [item["sentence1"] for item in batch]
#             positive_texts = [item["sentence2"] for item in batch]
#             contexts = [item["context"] for item in batch]

#             anchor_emb = model(anchor_texts, contexts)
#             positive_emb = model(positive_texts, contexts)
#             loss = contrastive_loss(anchor_emb, positive_emb)

#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#             epoch_loss += loss.item()
#             progress_bar.set_postfix({"batch_loss": loss.item()})

#         avg_loss = epoch_loss / len(train_loader)
#         print(f"Avg Training Loss: {avg_loss:.4f}")
#         val_loss = evaluate(model, val_loader)
#         print(f"Validation Loss: {val_loss:.4f}")

#     show_similarity(model, sentence1, sentence2, "the color of the animal")
#     show_similarity(model, sentence1, sentence2, "the type of animal")
    
#     model_path = "contextual_contrastive_model.pt"
#     torch.save(model.state_dict(), model_path)
#     print(f"\n Model weights saved to: {model_path}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModel
import random
import sys
import os
from tqdm import tqdm
from dataloader import ContrastiveBatchDataset, FixedBatchDataset
from train_plot import plot_loss_curves  # External plotting utility

# Model Definition
class ContextualContrastiveModel(nn.Module):
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2", proj_dim=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        self.context_mapper = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.context_gate = nn.Linear(hidden_size, hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim)
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_text(self, texts):
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.encoder.device)
        output = self.encoder(**encoded)
        return self.mean_pooling(output, encoded["attention_mask"])

    def forward(self, sentences, contexts):
        sent_emb = self.encode_text(sentences)
        ctx_emb = self.encode_text(contexts)
        gate = torch.sigmoid(self.context_gate(ctx_emb))
        mod_emb = (1 - gate) * sent_emb + gate * self.context_mapper(ctx_emb)
        mod_emb = F.normalize(mod_emb, dim=1)
        return self.projection(mod_emb)

# Contrastive Loss
def contrastive_loss(anchor_emb, positive_emb, temperature=0.1):
    anchor = F.normalize(anchor_emb, dim=1)
    positive = F.normalize(positive_emb, dim=1)
    logits = torch.matmul(anchor, positive.T) / temperature
    labels = torch.arange(len(anchor)).to(anchor.device)
    loss_a_to_p = F.cross_entropy(logits, labels)
    loss_p_to_a = F.cross_entropy(logits.T, labels)
    return (loss_a_to_p + loss_p_to_a) / 2

# Evaluation
def evaluate(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            anchor_texts = [item["sentence1"] for item in batch]
            positive_texts = [item["sentence2"] for item in batch]
            contexts = [item["context"] for item in batch]
            anchor_emb = model(anchor_texts, contexts)
            positive_emb = model(positive_texts, contexts)
            loss = contrastive_loss(anchor_emb, positive_emb)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# Train + Save Model
def train_and_evaluate(model_name, model, train_loader, val_loader, num_epochs=300, save_path_prefix="saved_models/300_contextual_contrastive_"):
    print(f"\n🔧 Training model with encoder: {model_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}", unit="batch")
        for batch in progress_bar:
            anchor_texts = [item["sentence1"] for item in batch]
            positive_texts = [item["sentence2"] for item in batch]
            contexts = [item["context"] for item in batch]

            anchor_emb = model(anchor_texts, contexts)
            positive_emb = model(positive_texts, contexts)
            loss = contrastive_loss(anchor_emb, positive_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": loss.item()})

        avg_train = epoch_loss / len(train_loader)
        avg_val = evaluate(model, val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")

    # Save model weights
    os.makedirs("saved_models", exist_ok=True)
    model_path = f"{save_path_prefix}{model_name.replace(' ', '_').lower()}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model weights saved to: {model_path}")

    return train_losses, val_losses

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_plot.py <embedding_file.pt>")
        sys.exit(1)

    embedding_file = sys.argv[1]
    batch_size = 256

    current_dataset = ContrastiveBatchDataset(embedding_file, batch_size=batch_size, threshold=0.8)

    def contrastive_collate_fn(_):
        return current_dataset.get_clean_batch()

    total_len = len(current_dataset.entries)
    indices = list(range(total_len))
    random.shuffle(indices)
    split = int(0.75 * total_len)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = Subset(current_dataset, train_indices)
    val_dataset = Subset(current_dataset, val_indices)

    train_batches = len(train_indices) // batch_size
    val_batches = len(val_indices) // batch_size

    train_loader = DataLoader(FixedBatchDataset(train_dataset, train_batches), batch_size=1, collate_fn=contrastive_collate_fn, shuffle=False)
    val_loader = DataLoader(FixedBatchDataset(val_dataset, val_batches), batch_size=1, collate_fn=contrastive_collate_fn, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    encoders = {
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "BERT-base": "bert-base-uncased",
        "SBERT": "sentence-transformers/paraphrase-MiniLM-L6-v2"
    }

    results = {}
    for name, enc in encoders.items():
        model = ContextualContrastiveModel(encoder_name=enc).to(device)
        train_loss, val_loss = train_and_evaluate(name, model, train_loader, val_loader, num_epochs=300)
        results[name] = (train_loss, val_loss)

    plot_loss_curves(results)