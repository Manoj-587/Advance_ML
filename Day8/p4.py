import pandas as pd
import sys, os

from ML_Modules import evaluate_min_samples

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
]

evaluate_min_samples(df, eps=2)
