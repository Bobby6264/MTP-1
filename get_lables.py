import pandas as pd

def main():
    print("Loading local clinical labels...")
    clinical_df = pd.read_csv("Data/combined_study_clinical_data.tsv", sep='\t')
    
    # Drop duplicate or empty sample IDs if present
    clinical_df = clinical_df.drop_duplicates(subset=['Sample ID']).dropna(subset=['Sample ID'])
    clinical_df.set_index('Sample ID', inplace=True)
    
    print("Loading preprocessed sample IDs...")
    gene_df = pd.read_csv("Data/processed_gene.csv", usecols=[0], index_col=0)
    my_samples = gene_df.index
    
    print(f"Aligning {len(my_samples)} preprocessed GIAC samples with clinical records...")
    
    # Reindex against our preprocessed cohort
    raw_subtypes = clinical_df['Subtype'].reindex(my_samples)
    
    # Robust string parser for subtypes
    def parse_subtype(val):
        if pd.isna(val):
            return "UNKNOWN"
        val_str = str(val).strip()
        if '_' in val_str:
            return val_str.split('_')[-1]
        return val_str

    clean_labels = raw_subtypes.apply(parse_subtype)
    
    # Filter strictly for the 5 MoXGATE target classes
    valid_classes = ['CIN', 'GS', 'MSI', 'HM-SNV', 'EBV']
    final_labels = clean_labels[clean_labels.isin(valid_classes)]
    
    print(f"\nSuccessfully mapped true molecular subtypes for {len(final_labels)} samples.")
    print("\nClass Distribution:")
    print(final_labels.value_counts())
    
    # Save the final mapped targets for training
    final_labels.to_frame(name="Subtype").to_csv("Data/true_labels.csv")
    print("\nSaved true_labels.csv successfully!")

if __name__ == "__main__":
    main()