import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


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


def evaluate_min_samples(df, eps=2):

    df_clean = treat_outliers(df)

    if "Detergents_Paper" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Detergents_Paper"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    for min_samples in [3, 4, 5]:

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        transformed = []
        for lbl in labels:
            if lbl == -1:
                transformed.append(1)
            else:
                transformed.append(lbl + 2)

        result = []
        for cid in sorted(set(transformed)):
            result.append((cid, transformed.count(cid)))

        print(f"eps = {eps} | min_samples = {min_samples} | obtained clustering: {result}")
