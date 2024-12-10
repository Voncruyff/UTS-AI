import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
Y = np.array([3, 6, 9, 12, 15])

# Model Regresi Linear
model = LinearRegression()
model.fit(X, Y)

# Prediksi untuk X = 6
X_new = np.array([[6]])
Y_pred = model.predict(X_new)

print(f"Prediksi untuk X = 6: {Y_pred[0]}")
