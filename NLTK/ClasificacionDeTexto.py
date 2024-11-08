import nltk
import random
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier, classify

# Descargar los datos necesarios de NLTK
nltk.download('punkt')

# Ejemplo de conjunto de datos de textos etiquetados
data = [
    ("I love this movie", "positive"),
    ("This movie is terrible", "negative"),
    ("This movie is great", "positive"),
    ("I dislike this movie", "negative"),
    ("This film is amazing", "positive"),
    ("I can't stand watching this movie", "negative"),
    ("The acting in this movie is phenomenal", "positive"),
    ("I regret wasting my time on this film", "negative"),
    ("I thoroughly enjoyed this movie", "positive"),
    ("This movie lacks depth and substance", "negative"),
    ("The plot of this movie was captivating", "positive"),
    ("I found the characters in this film to be very engaging", "positive"),
    ("The special effects in this movie were impressive", "positive"),
    ("The storyline was predictable and unoriginal", "negative"),
    ("I was disappointed by the lack of character development", "negative"),
    ("The cinematography in this film was stunning", "positive"),
    ("The dialogue felt forced and unnatural", "negative"),
    ("The pacing of the movie was too slow for my liking", "negative"),
    ("I was pleasantly surprised by how much I enjoyed this film", "positive"),
    ("The ending left me feeling unsatisfied and confused", "negative"),
    ("This movie exceeded my expectations", "positive"),
    ("The performances by the actors were lackluster", "negative")
]

# Preprocesamiento de datos: tokenización y extracción de características
def preprocess(text):
    tokens = word_tokenize(text)
    return {word: True for word in tokens}

# Aplicamos el preprocesamiento a los datos
featuresets = [(preprocess(text), label) for (text, label) in data]

# Dividimos los datos en conjuntos de entrenamiento y prueba
train_set, test_set = featuresets[:16], featuresets[16:]

# Entrenamos un clasificador utilizando Naive Bayes
classifier = NaiveBayesClassifier.train(train_set)

# Evaluamos el clasificador en el conjunto de prueba
accuracy = classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Clasificamos un nuevo texto
new_text = "This movie is amazing"
new_text_features = preprocess(new_text)
predicted_label = classifier.classify(new_text_features)
print("Predicted label:", predicted_label)
