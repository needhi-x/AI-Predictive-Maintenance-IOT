from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model

# Step 1: Load data
df = load_data()

# Step 2: Preprocess
df = preprocess_data(df)

# Step 3: Split data
X = df.drop("label", axis=1)
y = df["label"]

# Step 4: Train model
model = train_model(X, y)

# Step 5: Predict
predictions = model.predict(X)

print("\n🔍 Predictions:")
print(predictions)