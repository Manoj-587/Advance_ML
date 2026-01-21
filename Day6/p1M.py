import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

def check_correlation_text(input_df):
    corr_matrix = abs(input_df.corr()) >= 0.7
    print(corr_matrix)

def data_scale(X_DT):
    scaler = StandardScaler()
    numeric_cols = X_DT.select_dtypes(include='number').columns
    scaled_data = scaler.fit_transform(X_DT[numeric_cols])
    return pd.DataFrame(scaled_data, columns=numeric_cols)
