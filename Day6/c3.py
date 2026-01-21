import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import model_adaboost_classifier

filename = input().strip()

df = pd.read_csv(os.path.join(sys.path[0], filename))

print(df.head())

X = df.drop("price_range.enc", axis=1).values
y = df["price_range.enc"].values

model_adaboost_classifier(X, y)
