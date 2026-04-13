def create_features(df):
    df['temp_mean'] = df['temperature'].rolling(3).mean()
    df['vibration_mean'] = df['vibration'].rolling(3).mean()
    df = df.dropna()
    return df