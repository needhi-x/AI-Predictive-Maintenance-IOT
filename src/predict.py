def predict_failure(model, X):
    predictions = model.predict(X)
    print("\nPredictions:", predictions)
    return predictions