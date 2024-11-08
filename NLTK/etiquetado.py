import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger')

sentence = "NLTK es una biblioteca de procesamiento de lenguaje natural."
tokens = word_tokenize(sentence)
tagged_words = pos_tag(tokens)
print(tagged_words)

