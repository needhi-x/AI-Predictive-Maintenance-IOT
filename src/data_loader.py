import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv("sensor_data.csv")

    X = data[['temperature', 'pressure', 'vibration']]
    y = data['failure']

    return train_test_split(X, y, test_size=0.2, random_state=42)