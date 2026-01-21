import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import model_dbscan_cluster


filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit()

print(df.head())
print()

df.info()
print()

X = df.values

labels, sil, ch, dbi = model_dbscan_cluster(X)

df["cluster"] = labels

print(df["cluster"].value_counts())
print()

print(f"The average silhouette_score is: {sil:.2f}\n")
print(f"Calinski-Harabasz Index: {ch:.2f}\n")
print(f"Davies-Bouldin Index: {dbi:.2f}")
