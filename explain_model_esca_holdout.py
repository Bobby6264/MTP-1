import torch
import pandas as pd
import numpy as np
import sys

try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("Error: The 'captum' library is required. pip install captum")
    sys.exit(1)

# Import the model and load_data function specifically from the ESCA script
from train_esca_holdout import MoXGATE, load_data

def get_top_features(attributions, feature_names, top_n=10):
    """Helper function to get top N features based on absolute attribution scores"""
    mean_attrs = np.mean(np.abs(attributions), axis=0)
    top_indices = np.argsort(mean_attrs)[-top_n:][::-1]
    return [(feature_names[i], mean_attrs[i]) for i in top_indices]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data using the ESCA holdout script's function
    print("Loading data...")
    gene_df, mirna_df, meth_df, labels_df, train_samples, esca_samples, idx_to_class = load_data()
    
    # We want to explain the predictions specifically on the UNSEEN ESCA holdout samples
    # to see what features it relies on when generalizing to new data!
    print(f"\nExtracting the {len(esca_samples)} unseen ESCA holdout samples for explanation...")
    
    # Get the raw numpy arrays for the holdout set
    gene_X = gene_df.loc[esca_samples].values
    mirna_X = mirna_df.loc[esca_samples].values
    meth_X = meth_df.loc[esca_samples].values
    
    # Get the feature names
    gene_cols = gene_df.columns.tolist()
    mirna_cols = mirna_df.columns.tolist()
    meth_cols = meth_df.columns.tolist()

    # 2. Load the trained ESCA holdout model
    model = MoXGATE(
        gene_dim=gene_X.shape[1],
        mirna_dim=mirna_X.shape[1],
        meth_dim=meth_X.shape[1],
        num_classes=len(idx_to_class)
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("pth files/moxgate_esca_weights.pth", map_location=device))
        print("Successfully loaded pth files/moxgate_esca_weights.pth")
    except Exception as e:
        print("Could not load ESCA model weights! Have you trained it yet?")
        sys.exit(1)
        
    model.eval()

    # 3. Setup Integrated Gradients
    ig = IntegratedGradients(model)

    # Convert holdout samples to tensors
    t_gene = torch.tensor(gene_X, dtype=torch.float32, requires_grad=True).to(device)
    t_mirna = torch.tensor(mirna_X, dtype=torch.float32, requires_grad=True).to(device)
    t_meth = torch.tensor(meth_X, dtype=torch.float32, requires_grad=True).to(device)
    
    # Get model's predictions on the holdout set
    with torch.no_grad():
        preds = model(t_gene, t_mirna, t_meth).argmax(dim=1)

    print("\nCalculating Feature Attributions on the ESCA holdout set...")
    
    # Calculate attributions based on what the model predicted
    attributions = ig.attribute(inputs=(t_gene, t_mirna, t_meth), target=preds)
    
    attr_gene = attributions[0].cpu().detach().numpy()
    attr_mirna = attributions[1].cpu().detach().numpy()
    attr_meth = attributions[2].cpu().detach().numpy()

    # 4. Extract and print top features
    print("\n" + "="*50)
    print("TOP 10 FEATURES USED FOR PREDICTING UNSEEN ESCA SAMPLES")
    print("="*50)
    
    print("\n--- Top Gene Expression Features ---")
    for feat, score in get_top_features(attr_gene, gene_cols):
        print(f"{feat:<15}: {score:.6f}")
        
    print("\n--- Top miRNA Features ---")
    for feat, score in get_top_features(attr_mirna, mirna_cols):
        print(f"{feat:<15}: {score:.6f}")
        
    print("\n--- Top Methylation Features ---")
    for feat, score in get_top_features(attr_meth, meth_cols):
        print(f"{feat:<15}: {score:.6f}")

if __name__ == "__main__":
    main()
