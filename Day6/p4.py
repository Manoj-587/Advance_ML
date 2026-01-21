import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import data_scale, model_stacking_classifier


filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

X_df = df.loc[:, df.columns != "price_range"]
y = df["price_range"].values

X_scaled = data_scale(X_df)

X = X_scaled.values

model_stacking_classifier(X, y)
