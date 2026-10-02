import pandas as pd
import numpy as np

def apply_moxgate_preprocessing(df, modality_name):
    """
    1. Eliminate features with > 40% missing values[cite: 1]
    2. Median imputation for remaining missing values[cite: 1]
    """
    print(f"[{modality_name}] Original shape: {df.shape}")
    missing_pct = df.isnull().mean()
    
    df_filtered = df.loc[:, missing_pct <= 0.40]
    print(f"[{modality_name}] Features remaining after 40% threshold: {df_filtered.shape[1]}")
    
    df_imputed = df_filtered.fillna(df_filtered.median())
    print(f"[{modality_name}] Imputation complete.")
    return df_imputed

def main():
    print("Fetching GIAC clinical sample mappings from Pan-Cancer phenotype...")
    url = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz"
    clinical_df = pd.read_csv(url, sep='\t', compression='gzip', index_col=0)
    
    # We now know the exact column name from your log
    cancer_col = '_primary_disease'
    print(f"Using column '{cancer_col}' for cancer types.")
    
    # Match any rows containing the target GIAC cancer names (case-insensitive)
    keywords = ['colon', 'rectum', 'stomach', 'esophageal', 'coad', 'read', 'stad', 'esca']
    
    # Create a filter mask
    mask = clinical_df[cancer_col].astype(str).str.lower().apply(lambda x: any(k in x for k in keywords))
    giac_samples = clinical_df[mask].index.tolist()
    
    print(f"Identified {len(giac_samples)} GIAC samples in clinical data.")

    # 1. Process DNA Methylation (Memory Efficient Loading)
    print("\nReading Methylation headers...")
    meth_path = "Data/jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv"
    meth_headers = pd.read_csv(meth_path, sep="\t", nrows=0).columns.tolist()
    
    valid_meth_cols = [meth_headers[0]] + [col for col in meth_headers if col in giac_samples or col[:15] in giac_samples]
    print(f"Extracting {len(valid_meth_cols)-1} GIAC sample columns from DNA Methylation...")
    meth_df = pd.read_csv(meth_path, sep="\t", usecols=valid_meth_cols, index_col=0).T
    meth_processed = apply_moxgate_preprocessing(meth_df, "DNA Methylation")

    # 2. Process Gene Expression
    print("\nLoading Gene Expression...")
    gene_path = "Data/gene_expression.tsv.gz"
    gene_headers = pd.read_csv(gene_path, sep="\t", nrows=0).columns.tolist()
    valid_gene_cols = [gene_headers[0]] + [col for col in gene_headers if col in giac_samples or col[:15] in giac_samples]
    gene_df = pd.read_csv(gene_path, sep="\t", usecols=valid_gene_cols, index_col=0).T
    gene_processed = apply_moxgate_preprocessing(gene_df, "Gene Expression")

    # 3. Process miRNA Expression
    print("\nLoading miRNA Expression...")
    mirna_path = "Data/miRNA_expression.tsv.gz"
    try:
        mirna_df = pd.read_csv(mirna_path, sep="\t", index_col=0).T
    except Exception:
        mirna_df = pd.read_csv(mirna_path, sep=",", index_col=0).T
        
    mirna_df = mirna_df[mirna_df.index.isin(giac_samples) | mirna_df.index.str[:15].isin(giac_samples)]
    mirna_processed = apply_moxgate_preprocessing(mirna_df, "miRNA")

    # 4. Final Sample Alignment
    print("\nAligning sample IDs across all three modalities...")
    gene_processed.index = gene_processed.index.str[:15]
    mirna_processed.index = mirna_processed.index.str[:15]
    meth_processed.index = meth_processed.index.str[:15]

    common_samples = gene_processed.index.intersection(mirna_processed.index).intersection(meth_processed.index).unique()
    print(f"Final aligned sample count across all modalities: {len(common_samples)}")
    
    gene_final = gene_processed.loc[common_samples]
    mirna_final = mirna_processed.loc[common_samples]
    meth_final = meth_processed.loc[common_samples]
    
    # Save the processed matrices
    print("\nSaving processed matrices to Data/ folder...")
    gene_final.to_csv("Data/processed_gene.csv")
    mirna_final.to_csv("Data/processed_mirna.csv")
    meth_final.to_csv("Data/processed_methylation.csv")
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()