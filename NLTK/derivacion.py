from nltk.stem import PorterStemmer

words = ["running", "plays", "jumped"]
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in words]
print(stems)
