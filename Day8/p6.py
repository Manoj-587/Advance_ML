import pandas as pd
import sys, os

from ML_Modules import evaluate_dbscan

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit()

df = df[
    [
        "Channel",
        "Region",
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen"
    ]
].round(2)


sil, ch, dbi = evaluate_dbscan(df)

print(f"The average silhouette_score is: {sil:.2f}")
print(f"Calinski-Harabasz Index: {ch:.2f}")
print(f"Davies-Bouldin Index: {dbi:.2f}")
