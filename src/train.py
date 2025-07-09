import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load data
df = pd.read_csv('data/Iris.csv')
X = df.drop('Species', axis=1)
y = df['Species']

# Train model
model = KNeighborsClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model/model.pkl')
print(" Iris flower model trained.")
