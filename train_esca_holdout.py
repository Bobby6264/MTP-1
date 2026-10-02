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
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
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

        self.gate_weights = nn.Parameter(torch.tensor([0.333, 0.333, 0.333]))
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.norm_fusion = nn.LayerNorm(embed_dim)

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

        w = F.softmax(self.gate_weights, dim=0)
        tokens = torch.stack([w[0] * h_g, w[1] * h_m, w[2] * h_me], dim=1)

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

    unique_classes = sorted(labels_df['Subtype'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    labels_df['Subtype_Idx'] = labels_df['Subtype'].map(class_to_idx)

    # Known TCGA TSS codes for Esophageal Carcinoma (ESCA)
    esca_tss_codes = ['2H', '2M', 'IG', 'L5', 'L7', 'LN', 'M9', 'VR', 'V8', 'V9', 'IC', 'JY', 'P3', 'X8', 'S0', 'X9', 'Z6', 'ZA']
    
    # Extract TSS code from barcode and flag ESCA samples
    labels_df['TSS'] = labels_df.index.str.split('-').str[1]
    is_esca = labels_df['TSS'].isin(esca_tss_codes)
    
    esca_samples = labels_df[is_esca].index.tolist()
    train_samples = labels_df[~is_esca].index.tolist()
    
    print(f"Mapped Classes: {class_to_idx}")
    print(f"Found {len(train_samples)} training/val samples (COAD, READ, STAD)")
    print(f"Found {len(esca_samples)} holdout test samples (ESCA)")
    
    return gene_df, mirna_df, meth_df, labels_df, train_samples, esca_samples, idx_to_class

# ----------------- TRAINING LOOP -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gene_df, mirna_df, meth_df, labels_df, train_samples, esca_samples, idx_to_class = load_data()

    # Split non-ESCA data into Train (90%) and Validation (10%)
    train_y = labels_df.loc[train_samples, 'Subtype_Idx'].values
    train_idx, val_idx = train_test_split(train_samples, test_size=0.1, stratify=train_y, random_state=42)

    train_ds = MultiOmicsDataset(gene_df.loc[train_idx].values, mirna_df.loc[train_idx].values, meth_df.loc[train_idx].values, labels_df.loc[train_idx, 'Subtype_Idx'].values)
    val_ds = MultiOmicsDataset(gene_df.loc[val_idx].values, mirna_df.loc[val_idx].values, meth_df.loc[val_idx].values, labels_df.loc[val_idx, 'Subtype_Idx'].values)
    test_ds = MultiOmicsDataset(gene_df.loc[esca_samples].values, mirna_df.loc[esca_samples].values, meth_df.loc[esca_samples].values, labels_df.loc[esca_samples, 'Subtype_Idx'].values)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = MoXGATE(
        gene_dim=gene_df.shape[1],
        mirna_dim=mirna_df.shape[1],
        meth_dim=meth_df.shape[1],
        num_classes=len(idx_to_class)
    ).to(device)

    # Class weighting for Focal Loss based on training set distribution
    class_counts = np.bincount(labels_df.loc[train_idx, 'Subtype_Idx'].values, minlength=len(idx_to_class))
    # Prevent division by zero if a class is entirely missing in training data
    class_counts = np.where(class_counts == 0, 1, class_counts) 
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    weights = weights / weights.sum() * len(class_counts)
    weights = weights.to(device)
    
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_val_acc = 0.0
    epochs = 100

    print("\n--- Starting Training (Leave-One-Cancer-Out) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for g, m, me, label in train_loader:
            g, m, me, label = g.to(device), m.to(device), me.to(device), label.to(device)

            optimizer.zero_grad()
            out = model(g, m, me)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label)
            correct += (out.argmax(dim=1) == label).sum().item()
            total += len(label)

        train_acc = correct / total

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
            torch.save(model.state_dict(), "moxgate_esca_weights.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    print("\n--- Final ESCA Holdout Evaluation ---")
    model.load_state_dict(torch.load("moxgate_esca_weights.pth"))
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
    print(f"ESCA Test Accuracy: {test_acc*100:.2f}%")
    print(f"ESCA Test Weighted F1-Score: {test_f1:.4f}\n")
    
    # Identify which classes actually exist in the ESCA holdout
    present_classes = sorted(list(set(all_targets)))
    target_names = [idx_to_class[i] for i in present_classes]
    print(classification_report(all_targets, all_preds, labels=present_classes, target_names=target_names))

if __name__ == "__main__":
    main()