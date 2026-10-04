import torch
import pandas as pd
import numpy as np
import sys

# Try to import captum, warn if not installed
try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("Error: The 'captum' library is required for explainability.")
    print("Please install it by running: pip install captum")
    sys.exit(1)

# Import your model architecture from train.py
from train import MoXGATE, load_data

def get_top_features(attributions, feature_names, top_n=10):
    """Helper function to get top N features based on absolute attribution scores"""
    # Average attributions across all samples (absolute values)
    mean_attrs = np.mean(np.abs(attributions), axis=0)
    
    # Get indices of top N
    top_indices = np.argsort(mean_attrs)[-top_n:][::-1]
    
    return [(feature_names[i], mean_attrs[i]) for i in top_indices]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    gene_X, mirna_X, meth_X, y, class_names = load_data()
    
    # We need the original feature names (column names) from the CSVs
    print("Loading feature names...")
    gene_cols = pd.read_csv("Data/processed_gene.csv", nrows=0).columns.tolist()[1:]
    mirna_cols = pd.read_csv("Data/processed_mirna.csv", nrows=0).columns.tolist()[1:]
    meth_cols = pd.read_csv("Data/processed_methylation.csv", nrows=0).columns.tolist()[1:]

    # 2. Load the trained model
    model = MoXGATE(
        gene_dim=gene_X.shape[1],
        mirna_dim=mirna_X.shape[1],
        meth_dim=meth_X.shape[1],
        num_classes=len(class_names)
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("pth files/moxgate_best_weights.pth", map_location=device))
        print("Successfully loaded pth files/moxgate_best_weights.pth")
    except Exception as e:
        print("Could not load model weights! Have you trained the model yet?")
        sys.exit(1)
        
    model.eval()

    # 3. Setup Integrated Gradients
    ig = IntegratedGradients(model)

    # Stratified sampling of 50 samples for explainability
    from sklearn.model_selection import train_test_split
    
    # We use y (true labels) to stratify the sample evenly across cancer subtypes
    idx = np.arange(len(y))
    _, strat_idx = train_test_split(idx, test_size=50, stratify=y, random_state=42)
    
    t_gene = torch.tensor(gene_X[strat_idx], dtype=torch.float32, requires_grad=True).to(device)
    t_mirna = torch.tensor(mirna_X[strat_idx], dtype=torch.float32, requires_grad=True).to(device)
    t_meth = torch.tensor(meth_X[strat_idx], dtype=torch.float32, requires_grad=True).to(device)
    
    # The actual predictions for these samples
    with torch.no_grad():
        preds = model(t_gene, t_mirna, t_meth).argmax(dim=1)

    print("\nCalculating Feature Attributions... (This may take a minute)")
    
    # Calculate attributions for each sample based on its predicted class
    # We pass the inputs as a tuple and specify the target class for each sample
    attributions = ig.attribute(inputs=(t_gene, t_mirna, t_meth), target=preds)
    
    attr_gene = attributions[0].cpu().detach().numpy()
    attr_mirna = attributions[1].cpu().detach().numpy()
    attr_meth = attributions[2].cpu().detach().numpy()

    # 4. Extract and print top features
    print("\n" + "="*50)
    print("TOP 10 MOST IMPORTANT BIOLOGICAL FEATURES")
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
