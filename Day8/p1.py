import pandas as pd
import sys, os

from ML_Modules import (
    assess_outliers,
    check_multicollinearity,
    remove_redundant_features,
    scale_data
)

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit()

print("Dataset Preview:")
print(df.head())
print()

print("Dataset Info:")
info = df.info()
print(info)
print()

print("Dataset Description:")
desc = df.describe()
print(desc)

print("Missing Values:")
missing = df.isnull().sum()
print(missing)


df_outlier = assess_outliers(df)

print("Data After Outlier Treatment:")
print(df_outlier.head())


print("Multicollinearity Matrix:")
multi = check_multicollinearity(df_outlier)
print(multi)


df_reduced = remove_redundant_features(df_outlier)

print("Columns after removal:")
print(list(df_reduced.columns))
print()

scaled_df = scale_data(df_reduced)

print("Scaled Data Preview:")
print(scaled_df.head())
