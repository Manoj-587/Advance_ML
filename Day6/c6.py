import pandas as pd
import sys,os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import (
    model_adaboost_regressor,
    model_stacking_regressor
)

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

if "price" not in df.columns:
    print("Error: price column not found.")
    sys.exit(1)

X = df.drop("price", axis=1).values
y = df["price"].values

print("=== AdaBoost Regressor Performance ===")
model_adaboost_regressor(X, y)

print("\n=== Stacking Regressor Performance ===")
model_stacking_regressor(X, y)
