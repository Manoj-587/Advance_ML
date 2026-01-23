import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Import custom module
from ML_Modules import scale, evaluate_clusterer

# Sklearn imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ---------------- LOAD DATA ----------------
def load_data():
    file = input("Enter your data file (CSV or XLSX): ")

    path = os.path.join(sys.path[0], file)

    if not os.path.exists(path):
        print("File not found!")
        exit()

    if file.endswith(".csv"):
        return pd.read_csv(path)
    elif file.endswith(".xlsx") or file.endswith(".xls"):
        return pd.read_excel(path)
    else:
        print("Invalid file format!")
        exit()


# ---------------- CLUSTERING FUNCTION ----------------
def clustering(X, is_pca=False):

    if is_pca:
        print("\nSilhouette Scores WITH PCA:")
    else:
        print("\nSilhouette Scores WITHOUT PCA:")

    # Silhouette scores for k=2 to 9
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=10)
        labels = km.fit_predict(X)
        sil = round(silhouette_score(X, labels), 3)
        print(f"k={k}: Silhouette Score = {sil}")

    # Run final clustering with k=2
    if is_pca:
        print("\nRunning KMeans WITH PCA...")
    else:
        print("\nRunning KMeans WITHOUT PCA...")

    km = KMeans(n_clusters=2, random_state=10)
    labels = km.fit_predict(X)

    evaluate_clusterer(X, labels)


# ---------------- MAIN FUNCTION ----------------
def main():

    # Load dataset
    df = load_data()
    print("\nDataset Loaded Successfully!")
    print(df.head())
    print(df.info())

    # Separate input and output
    X = df[["age", "children", "charges", "gender_n", "smoker_n", "region_n"]]
    y = df["weight_condition_n"]

    print("\nInput Data:")
    print(X.head())

    print("\nOutput Data:")
    print(y.head())

    # Scaling
    print("\nScaling Input Data...")
    X_scaled = scale(X)

    # ----------- WITHOUT PCA -----------
    clustering(X_scaled, is_pca=False)

    # ----------- PCA -----------
    print("\nRunning PCA (n_components=2)...")
    pca = PCA(n_components=2, random_state=10)
    X_pca = pca.fit_transform(X_scaled)

    clustering(X_pca, is_pca=True)



    # Final Summary
    print("\n==================== SUMMARY ====================")
    print("Data Loaded")
    print("Data Scaled")
    print("Optimal k checked using Silhouette Score")
    print("K-Means applied WITHOUT PCA")
    print("K-Means applied WITH PCA")
    print("Evaluation completed using silhouette score")
    print("==================================================")
    

# Run program
if __name__ == "__main__":
    main()