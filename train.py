import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

# ----------------- FOCAL LOSS -----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Calculate raw CE loss first to keep probabilities mathematically accurate
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply the focal scaling term
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        # Apply class weights safely after probability calculation
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        return focal_loss.mean()
# ----------------- MOXGATE ARCHITECTURE -----------------
class ModalityEncoder(nn.Module):
    def __init__(self, in_features, embed_dim=256, num_heads=8):
        super(ModalityEncoder, self).__init__()
        self.fc = nn.Linear(in_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        h = self.norm(F.relu(self.fc(x))).unsqueeze(1)
        attn_out, _ = self.self_attn(h, h, h)
        return attn_out.squeeze(1)

class MoXGATE(nn.Module):
    def __init__(self, gene_dim, mirna_dim, meth_dim, num_classes=4, embed_dim=256):
        super(MoXGATE, self).__init__()
        self.gene_enc = ModalityEncoder(gene_dim, embed_dim)
        self.mirna_enc = ModalityEncoder(mirna_dim, embed_dim)
        self.meth_enc = ModalityEncoder(meth_dim, embed_dim)

        # Learnable gating weights (initialized equally)
        self.gate_weights = nn.Parameter(torch.tensor([0.333, 0.333, 0.333]))

        # Cross-Attention Fusion (32 heads as per paper)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=32, batch_first=True)
        self.norm_fusion = nn.LayerNorm(embed_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, g, m, me):
        h_g = self.gene_enc(g)
        h_m = self.mirna_enc(m)
        h_me = self.meth_enc(me)

        # Gated combination
        w = F.softmax(self.gate_weights, dim=0)
        tokens = torch.stack([w[0] * h_g, w[1] * h_m, w[2] * h_me], dim=1)

        # Cross-Attention
        fused_tokens, _ = self.cross_attn(tokens, tokens, tokens)
        fused_rep = self.norm_fusion(fused_tokens.mean(dim=1))

        return self.classifier(fused_rep)

# ----------------- DATASET & DATALOADER -----------------
class MultiOmicsDataset(Dataset):
    def __init__(self, gene, mirna, meth, labels):
        self.gene = torch.tensor(gene, dtype=torch.float32)
        self.mirna = torch.tensor(mirna, dtype=torch.float32)
        self.meth = torch.tensor(meth, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.gene[idx], self.mirna[idx], self.meth[idx], self.labels[idx]

def load_data():
    print("Loading true labels...")
    labels_df = pd.read_csv("Data/true_labels.csv", index_col=0)
    samples = labels_df.index.tolist()

    print(f"Loading and filtering multi-omic features for {len(samples)} samples...")
    gene_df = pd.read_csv("Data/processed_gene.csv", index_col=0).loc[samples]
    mirna_df = pd.read_csv("Data/processed_mirna.csv", index_col=0).loc[samples]
    meth_df = pd.read_csv("Data/processed_methylation.csv", index_col=0).loc[samples]

    # Map labels to integers
    unique_classes = sorted(labels_df['Subtype'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    y = labels_df['Subtype'].map(class_to_idx).values

    print(f"Mapped Classes: {class_to_idx}")
    return gene_df.values, mirna_df.values, meth_df.values, y, idx_to_class

# ----------------- TRAINING LOOP -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gene_X, mirna_X, meth_X, y, class_names = load_data()

    # Stratified Train/Val/Test split (80% Train, 10% Val, 10% Test)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, stratify=y[test_idx], random_state=42)

    train_ds = MultiOmicsDataset(gene_X[train_idx], mirna_X[train_idx], meth_X[train_idx], y[train_idx])
    val_ds = MultiOmicsDataset(gene_X[val_idx], mirna_X[val_idx], meth_X[val_idx], y[val_idx])
    test_ds = MultiOmicsDataset(gene_X[test_idx], mirna_X[test_idx], meth_X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = MoXGATE(
        gene_dim=gene_X.shape[1],
        mirna_dim=mirna_X.shape[1],
        meth_dim=meth_X.shape[1],
        num_classes=len(class_names)
    ).to(device)

    # Class weighting for Focal Loss (Matched alpha=1 / None as per paper)
    criterion = FocalLoss(alpha=None, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    best_val_acc = 0.0
    epochs = 100

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for g, m, me, label in train_loader:
            g, m, me, label = g.to(device), m.to(device), me.to(device), label.to(device)

            optimizer.zero_grad()
            out = model(g, m, me)
            loss = criterion(out, label)
            
            # Add λ₁‖w−1‖² regularizer (λ₁ = 0.01)
            reg_loss = 0.01 * torch.sum((model.gate_weights - 1.0) ** 2)
            loss = loss + reg_loss
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label)
            correct += (out.argmax(dim=1) == label).sum().item()
            total += len(label)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for g, m, me, label in val_loader:
                g, m, me, label = g.to(device), m.to(device), me.to(device), label.to(device)
                out = model(g, m, me)
                val_correct += (out.argmax(dim=1) == label).sum().item()
                val_total += len(label)

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "pth files/moxgate_best_weights.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Evaluation on Test Set
    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load("pth files/moxgate_best_weights.pth"))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for g, m, me, label in test_loader:
            g, m, me = g.to(device), m.to(device), me.to(device)
            out = model(g, m, me)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_targets.extend(label.numpy())

    test_acc = accuracy_score(all_targets, all_preds)
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Weighted F1-Score: {test_f1:.4f}\n")
    print(classification_report(all_targets, all_preds, target_names=[class_names[i] for i in range(len(class_names))]))

if __name__ == "__main__":
    main()