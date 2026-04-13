import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Data Loaded:\n", df.head())
    return df

def preprocess(df):
    df = df.dropna()
    return df