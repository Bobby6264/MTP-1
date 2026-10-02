# import pandas as pd

# # 1. Load the 902 labels
# labels_df = pd.read_csv("Data/true_labels.csv", index_col=0)

# # 2. Extract the 2-character TSS code from the barcode (e.g., 'TCGA-3L-...' -> '3L')
# tss_codes = labels_df.index.str.split('-').str[1]

# # 3. Define the official TCGA mapping for these four cancers
# esca_codes = ['2H', '2M', 'IG', 'L5', 'L7', 'LN', 'M9', 'VR', 'V8', 'V9', 'IC', 'JY', 'P3', 'X8', 'S0', 'X9', 'Z6', 'ZA']
# stad_codes = ['B8', 'BR', 'CD', 'CG', 'D7', 'F1', 'FP', 'HF', 'HU', 'IN', 'IP', 'K9', 'L1', 'M7', 'MX', 'MQ', 'RD', 'SW', 'VQ', 'YY', 'ZQ']
# read_codes = ['AF', 'AG', 'AH', 'DC', 'DT', 'DY', 'EF', 'EI', 'F5', 'G5']
# coad_codes = ['3L', '4N', '4T', '5M', 'A6', 'AA', 'AD', 'AM', 'AU', 'AY', 'AZ', 'CA', 'CI', 'CK', 'CM', 'D5', 'DM', 'F4', 'G4', 'NH', 'QG', 'QL', 'RU', 'SS', 'WS']

# # 4. Calculate and print the totals
# coad_count = tss_codes.isin(coad_codes).sum()
# read_count = tss_codes.isin(read_codes).sum()
# stad_count = tss_codes.isin(stad_codes).sum()
# esca_count = tss_codes.isin(esca_codes).sum()

# print(f"COAD (Colon): {coad_count}")
# print(f"READ (Rectum): {read_count}")
# print(f"STAD (Stomach): {stad_count}")
# print(f"ESCA (Esophagus): {esca_count}")
# print("-" * 25)
# print(f"Total Samples: {coad_count + read_count + stad_count + esca_count}")

import pandas as pd

labels_df = pd.read_csv("Data/true_labels.csv", index_col=0)
tss_counts = labels_df.index.str.split('-').str[1].value_counts()

print(tss_counts)
print("-" * 25)
print(f"Total Samples: {tss_counts.sum()}")