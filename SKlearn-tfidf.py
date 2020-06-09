from sklearn.feature_extraction.text import TfidfVectorizer

sentence1 = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
sentence2 = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"

data = [sentence1, sentence2]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(data)
print(X)
