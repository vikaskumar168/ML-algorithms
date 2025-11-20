import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

print("Loading California housing dataset...")
# Load California housing dataset
data = fetch_california_housing()
# print(data)

dataset = pd.DataFrame(data.data, columns=data.feature_names)
# print(dataset)


# Separate features and target variable
X = dataset
y = data.target

# print(y)


#train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)
# print(X_train)

#Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# print(X_train)
X_test = scaler.transform(X_test)

#Model Training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")
# Make predictions
y_pred = model.predict(X_test)
print("Predictions made on test set.")

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Visualize the results
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Housing Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()  
print("Visualization completed.")

#More visualizations
residuals = y_test - y_pred
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()      


