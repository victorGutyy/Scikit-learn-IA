import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.metrics import precision_score, recall_score, f1_score

# Descargar los datos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Crear una lista de stopwords en español y un stemmer en español
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer("spanish")

# Conjunto de datos ampliado: cada entrada tiene comentario, sentimiento, género y recomendación de edad
data = [
    ("Me encanta esta película", "positivo", "romance", "para mayores de 13"),
    ("Esta película es terrible", "negativo", "acción", "para mayores de 18"),
    ("Es una gran película", "positivo", "drama", "para todo público"),
    ("No me gusta esta película", "negativo", "comedia", "para todo público"),
    ("Este filme es asombroso", "positivo", "acción", "para mayores de 13"),
    ("No soporto ver esta película", "negativo", "romance", "para mayores de 13"),
    ("La actuación en esta película es fenomenal", "positivo", "drama", "para mayores de 13"),
    ("Lamento haber perdido mi tiempo en esta película", "negativo", "comedia", "para todo público"),
    ("Disfruté mucho esta película", "positivo", "romance", "para mayores de 13"),
    ("Esta película carece de profundidad y sustancia", "negativo", "acción", "para mayores de 18"),
    ("La trama de esta película fue cautivante", "positivo", "drama", "para todo público"),
    ("Los personajes en esta película son muy atractivos", "positivo", "comedia", "para todo público"),
    ("Los efectos especiales en esta película fueron impresionantes", "positivo", "acción", "para mayores de 13"),
    ("La historia fue predecible y poco original", "negativo", "romance", "para mayores de 13"),
    ("Me decepcionó la falta de desarrollo de los personajes", "negativo", "drama", "para mayores de 13"),
    ("La cinematografía en este filme fue impresionante", "positivo", "drama", "para todo público"),
    ("Los diálogos se sintieron forzados y poco naturales", "negativo", "comedia", "para todo público"),
    ("El ritmo de la película fue demasiado lento para mi gusto", "negativo", "drama", "para mayores de 13"),
    ("Me sorprendió gratamente cuánto disfruté de este filme", "positivo", "romance", "para mayores de 13"),
    ("El final me dejó insatisfecho y confundido", "negativo", "acción", "para mayores de 18"),
    ("Esta película superó mis expectativas", "positivo", "comedia", "para todo público"),
    ("Las actuaciones de los actores fueron mediocres", "negativo", "romance", "para mayores de 13"),
    ("Es una obra maestra del cine moderno", "positivo", "drama", "para mayores de 13"),
    ("El argumento de la película es difícil de seguir", "negativo", "acción", "para mayores de 18"),
    ("La banda sonora es increíble", "positivo", "romance", "para todo público"),
    ("La película es demasiado larga y aburrida", "negativo", "comedia", "para todo público"),
    ("La actuación principal fue espectacular", "positivo", "drama", "para mayores de 13"),
    ("El guion de la película es débil", "negativo", "acción", "para mayores de 18"),
    ("Una película perfecta para disfrutar en familia", "positivo", "comedia", "para todo público"),
    ("Contiene mucha violencia para mi gusto", "negativo", "acción", "para mayores de 18")
]

# Preprocesamiento de datos: tokenización, eliminación de stopwords y stemming
def preprocess(text, remove_stopwords=True, apply_stemming=True):
    tokens = word_tokenize(text.lower())
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    if apply_stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    return {word: True for word in tokens}

# Crear conjuntos de características para cada clasificación
sentiment_features = [(preprocess(text), sentiment) for (text, sentiment, genre, age) in data]
genre_features = [(preprocess(text), genre) for (text, sentiment, genre, age) in data]
age_features = [(preprocess(text), age) for (text, sentiment, genre, age) in data]

# Dividir los datos en conjuntos de entrenamiento y prueba para cada clasificador
train_sentiment, test_sentiment = sentiment_features[:24], sentiment_features[24:]
train_genre, test_genre = genre_features[:24], genre_features[24:]
train_age, test_age = age_features[:24], age_features[24:]

# Entrenar un clasificador Naive Bayes para cada atributo
sentiment_classifier = NaiveBayesClassifier.train(train_sentiment)
genre_classifier = NaiveBayesClassifier.train(train_genre)
age_classifier = NaiveBayesClassifier.train(train_age)

# Evaluar la precisión de cada clasificador
print("Naive Bayes Accuracy (sentimiento):", accuracy(sentiment_classifier, test_sentiment))
print("Naive Bayes Accuracy (género):", accuracy(genre_classifier, test_genre))
print("Naive Bayes Accuracy (recomendación de edad):", accuracy(age_classifier, test_age))

# Función para clasificar nuevo texto de entrada del usuario
def classify_user_input(sentiment_classifier, genre_classifier, age_classifier):
    print("\n¡Bienvenido al clasificador de sentimientos, géneros y recomendaciones de edad!")
    print("Escribe un comentario sobre una película en español y el modelo intentará clasificar 'sentimiento', 'género' y 'recomendación de edad'.")
    print("Escribe 'salir' para terminar el programa.\n")
    
    while True:
        user_input = input("Ingrese un comentario sobre la película: ")
        if user_input.lower() == 'salir':
            print("Gracias por usar el clasificador de sentimientos, géneros y recomendaciones de edad. ¡Hasta luego!")
            break
        else:
            # Clasificar el comentario del usuario en sentimiento, género y recomendación de edad
            features = preprocess(user_input)
            predicted_sentiment = sentiment_classifier.classify(features)
            predicted_genre = genre_classifier.classify(features)
            predicted_age = age_classifier.classify(features)
            print(f"Predicted Sentiment: {predicted_sentiment}")
            print(f"Predicted Genre: {predicted_genre}")
            print(f"Predicted Age Recommendation: {predicted_age}\n")

# Iniciar interacción con el usuario
classify_user_input(sentiment_classifier, genre_classifier, age_classifier)
