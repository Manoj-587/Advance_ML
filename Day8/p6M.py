import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def treat_outliers(df):
    df_out = df.copy().astype(float)

    for col in df_out.columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_out[col] = np.where(
            df_out[col] < lower, lower,
            np.where(df_out[col] > upper, upper, df_out[col])
        )

    return df_out.round(2)



def evaluate_dbscan(df):

    df_clean = treat_outliers(df)

    if "Detergents_Paper" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Detergents_Paper"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    dbscan = DBSCAN(eps=2, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)

    return sil, ch, dbi
