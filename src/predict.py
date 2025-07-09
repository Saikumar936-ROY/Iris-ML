import pandas as pd
import joblib

# Load trained model
model = joblib.load('model/model.pkl')

# Sample flower data
sample = pd.DataFrame({
    'SepalLengthCm': [5.1],
    'SepalWidthCm': [3.5],
    'PetalLengthCm': [1.4],
    'PetalWidthCm': [0.2]
})

# Predict species
prediction = model.predict(sample)[0]
print(f"🌸 Predicted Iris Species: {prediction}")

