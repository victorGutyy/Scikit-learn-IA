from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# cargar el conjunto de datos california_housing 
California = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(California.data, California.target, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo 
model.fit(X_train, y_train)

# predecir los precios de las viviendas para los datos de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio:", mse)