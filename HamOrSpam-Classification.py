from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

mapping_dict = {'ham': 0, 'spam': 1}
inv_mapping_dict = {v: k for k, v in mapping_dict.items()}


def data_prep(df):
    X = df['text']
    y = df['type'].map(mapping_dict)
    return X,y

def vetorize_data(X,y):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(X.values.astype('U'))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return tfidf, X_train, X_test, y_train, y_test

def train_model(X_train,y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def accuracy_score(model,X_test,y_test):
    return model.score(X_test, y_test)

def predict(text,model,vectorizer_model):
    data = [text]
    vect = vectorizer_model.transform(data).toarray()
    my_prediction = model.predict(vect)
    return inv_mapping_dict[my_prediction[0]]

df = pd.read_csv("data/hamOrspam.csv", encoding="latin-1")
X,y = data_prep(df)

vectorizer_model, X_train, X_test, y_train, y_test = vetorize_data(X,y)

model = train_model(X_train,y_train)
print("Accuracy:",accuracy_score(model,X_test,y_test))

test_text = "You have won 1 Lakh lottery! please click the link below to claim the money"
print(predict(test_text,model,vectorizer_model))