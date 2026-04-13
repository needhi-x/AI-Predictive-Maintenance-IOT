from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model(df):
    X = df[['temperature','vibration','pressure','temp_mean','vibration_mean']]
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

    print("\nModel Evaluation:\n")
    print(classification_report(y_test, model.predict(X_test)))

    return model, X_test, y_test