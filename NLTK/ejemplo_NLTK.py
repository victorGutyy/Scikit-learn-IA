import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Descargar los datos necesarios de NLTK para el análisis de sentimiento
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Datos ficticios de muestra con comentarios añadidos
data = {
    "Departamento": ["Antioquia", "Boyacá", "Cundinamarca", "Nariño", "La Guajira", "Valle del Cauca", "Santander", "Cauca", "Magdalena", "Tolima"],
    "Temperatura_Promedio": [22.5, 16.2, 18.3, 14.0, 28.4, 24.1, 23.5, 19.2, 27.8, 25.4],  # en grados Celsius
    "Radiacion_Solar_Promedio": [4.5, 5.2, 4.8, 3.9, 6.1, 5.0, 5.3, 4.6, 6.0, 5.4],  # en kWh/m²/día
    "Porcentaje_Efectividad": [85, 78, 82, 70, 90, 83, 80, 75, 88, 84],  # Porcentaje de éxito en la implementación solar
    "Recomendacion_Cultivo": ["Café", "Papa", "Flores", "Caña de Azúcar", "Aguacate", "Frutas Tropicales", "Cacao", "Café", "Banano", "Maíz"],
    "Comentario": [
        "La energía solar ha reducido los costos en Antioquia de forma significativa.",
        "En Boyacá, los resultados no han sido tan efectivos como esperábamos.",
        "Cundinamarca ha tenido un impacto positivo con la energía solar en sus cultivos.",
        "La implementación en Nariño ha sido desafiante debido a la baja radiación solar.",
        "En La Guajira, el sistema solar ha funcionado excepcionalmente bien.",
        "En el Valle del Cauca, los beneficios han sido considerables y ha mejorado la producción.",
        "Santander ha visto una buena aceptación de la energía solar entre los agricultores.",
        "En Cauca, la energía solar ha ayudado pero podría ser más eficiente.",
        "La implementación de energía solar en Magdalena ha sido excelente.",
        "En Tolima, el sistema solar ha mejorado la rentabilidad de los cultivos."
    ]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Convertir variables categóricas en numéricas
df['Departamento'] = df['Departamento'].astype('category').cat.codes
df['Recomendacion_Cultivo'] = df['Recomendacion_Cultivo'].astype('category').cat.codes

# Variables predictoras (X) y variable objetivo (y)
X = df[["Temperatura_Promedio", "Radiacion_Solar_Promedio", "Departamento"]]
y = df["Porcentaje_Efectividad"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones y evaluación del modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Análisis de sentimiento para cada comentario
print("\nAnálisis de Sentimiento por Departamento:")
for i, row in df.iterrows():
    comentario = row['Comentario']
    score = sia.polarity_scores(comentario)
    print(f"Departamento: {row['Departamento']} | Comentario: {comentario}")
    print(f"Sentimiento: {score}\n")
