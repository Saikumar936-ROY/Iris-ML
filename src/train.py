import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Load data
df = pd.read_csv('data/Iris.csv')

# Drop unnecessary columns (e.g., Id)
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Train model
model = KNeighborsClassifier()
model.fit(X, y)

# Create model directory if not exists
os.makedirs('model', exist_ok=True)

# Save the model
joblib.dump(model, 'model/model.pkl')

print("✅ Iris flower classification model trained and saved to 'model/model.pkl'")
