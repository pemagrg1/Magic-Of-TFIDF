from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

text1 = "This is a foo bar sentence ."
text2 = "This sentence is similar to a foo bar sentence ."

tfidf = TfidfVectorizer(stop_words='english')
data = tfidf.fit_transform([text1,text2])

vector1 = tfidf.transform([text1]).toarray()
vector2 = tfidf.transform([text2]).toarray()

cosine = cosine_similarity(vector1, vector2)
print("Cosine:", cosine)