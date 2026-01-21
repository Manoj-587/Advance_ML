import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def assess_outliers(df):
    df_out = df.copy().astype(float)

    numeric_cols = df_out.select_dtypes(include="number").columns

    for col in numeric_cols:
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


def check_multicollinearity(df):
    corr = df.corr().abs()
    return corr >= 0.7


def remove_redundant_features(df):
    if "Detergents_Paper" in df.columns:
        df = df.drop(columns=["Detergents_Paper"])
    return df


def scale_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)
