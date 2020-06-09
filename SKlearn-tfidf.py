from sklearn.feature_extraction.text import TfidfVectorizer

first_sentence = "Data Science is the sexiest job of the 21st century"
second_sentence = "machine learning is the key for data science"

data = [first_sentence,second_sentence]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(data)
print(X)



