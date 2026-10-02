import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        # Linear transformation to align dimensions
        self.proj = nn.Linear(input_dim, embed_dim)
        self.activation = nn.ReLU()
        # Modality-specific Self-Attention (8 heads, 0.1 dropout)[cite: 1]
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1) 
        h = self.activation(self.proj(x))
        # Self-attention computation[cite: 1]
        attn_out, _ = self.self_attn(h, h, h)
        return self.norm(attn_out + h) # Residual connection

class MoXGATE(nn.Module):
    def __init__(self, gene_dim, meth_dim, mirna_dim, num_classes=5):
        super().__init__()
        embed_dim = 256
        
        # 1. Independent Encoders (Weights are not shared across modalities)[cite: 1]
        self.gene_encoder = ModalityEncoder(gene_dim, embed_dim, num_heads=8)
        self.meth_encoder = ModalityEncoder(meth_dim, embed_dim, num_heads=8)
        self.mirna_encoder = ModalityEncoder(mirna_dim, embed_dim, num_heads=8)
        
        # 2. Modality Importance Learning (Initialized at 0.33 each)[cite: 1]
        self.w = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]))
        
        # 3. Modality-Weighted Cross-Attention (32 heads, embed_dim 256)[cite: 1]
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=32, batch_first=True)
        
        # Feedforward Attention mechanism included, no BatchNorm[cite: 1]
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        
        # 4. Subtype Classifier Module[cite: 1]
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),     # Final embedding dimension of 128[cite: 1]
            nn.ReLU(),
            nn.Dropout(0.3),               # Dropout rate of 0.3[cite: 1]
            nn.Linear(128, num_classes)    # Output classes (CIN, GS, MSI, HM-SNV, EBV)[cite: 1]
        )

    def forward(self, x_gene, x_meth, x_mirna):
        # Process each modality
        z_gene = self.gene_encoder(x_gene)
        z_meth = self.meth_encoder(x_meth)
        z_mirna = self.mirna_encoder(x_mirna)
        
        # Ensure weights sum to 1 via softmax
        weights = F.softmax(self.w, dim=0)
        
        # Stack representations for cross-attention: shape (batch_size, 3, embed_dim)[cite: 1]
        C = torch.cat([z_gene, z_meth, z_mirna], dim=1)
        
        # Cross Attention (Query, Key, Value all derived from stacked representations)[cite: 1]
        F_attn, _ = self.cross_attn(C, C, C)
        F_attn = self.cross_norm(F_attn + C)
        F_ffn = self.ffn(F_attn)
        
        # Apply learnable modality weights[cite: 1]
        F_final = (weights[0] * F_ffn[:, 0, :]) + (weights[1] * F_ffn[:, 1, :]) + (weights[2] * F_ffn[:, 2, :])
        
        # Classification
        logits = self.classifier(F_final)
        return logits

if __name__ == "__main__":
    # Test model initialization with the dimensions from your preprocessing log
    gene_dim = 19076
    meth_dim = 22601
    mirna_dim = 743
    
    print("Initializing MoXGATE model...")
    model = MoXGATE(gene_dim, meth_dim, mirna_dim)
    
    # Create dummy tensors simulating a batch of 4 samples
    dummy_gene = torch.rand(4, gene_dim)
    dummy_meth = torch.rand(4, meth_dim)
    dummy_mirna = torch.rand(4, mirna_dim)
    
    print("Running forward pass test...")
    output = model(dummy_gene, dummy_meth, dummy_mirna)
    
    print(f"Output shape (Batch Size x Num Classes): {output.shape}")
    print("Model architecture successfully compiled!")