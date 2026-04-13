import pandas as pd
from src.data_preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("Loading data...")
    df = pd.read_csv("data/sensor_data.csv")

    print("Preprocessing...")
    df = preprocess(df)

    print("Preparing data...")
    X = df[['temperature', 'pressure', 'vibration']]
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Training model...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print("Evaluating...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.2f}")

    print("Saving model...")
    joblib.dump(model, "models/model.pkl")

    # -------- Manual Prediction --------
    print("\n--- Manual Prediction ---")
    temp = float(input("Temperature: "))
    pressure = float(input("Pressure: "))
    vibration = float(input("Vibration: "))

    new_data = pd.DataFrame([[temp, pressure, vibration]],
                            columns=['temperature','pressure','vibration'])

    result = model.predict(new_data)

    if result[0] == 1:
        print("⚠️ Machine Failure Likely!")
    else:
        print("✅ Machine is Normal")

if __name__ == "__main__":
    main()