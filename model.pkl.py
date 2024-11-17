import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Generate synthetic data
np.random.seed(42)
data = {
    'Size': np.random.randint(500, 4000, 100),
    'Bedrooms': np.random.randint(1, 5, 100),
    'Age': np.random.randint(1, 30, 100),
}
data['Price'] = data['Size'] * 300 + data['Bedrooms'] * 5000 - data['Age'] * 200 + np.random.randint(-10000, 10000, 100)
df = pd.DataFrame(data)

# Features and target
X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
