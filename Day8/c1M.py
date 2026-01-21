import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def model_dbscan_cluster(X):

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distance = distances[:, 1]

    eps = 2
    for min_samples in [3, 4, 5]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)

        transformed = []
        for l in labels:
            if l == -1:
                transformed.append(1)
            else:
                transformed.append(l + 2)

        result = []
        for cid in sorted(set(transformed)):
            result.append((cid, transformed.count(cid)))

        print(f"eps= {eps} | min_samples=  {min_samples} | obtained clustering:  {result}")

    print()

    final_db = DBSCAN(eps=2, min_samples=3)
    final_labels = final_db.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, final_labels)
    ch = calinski_harabasz_score(X_scaled, final_labels)
    dbi = davies_bouldin_score(X_scaled, final_labels)

    return final_labels, sil, ch, dbi
