"""
    HamOrSpam Classification using TFIDF
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

mapping_dict = {'ham': 0, 'spam': 1}
inv_mapping_dict = {v: k for k, v in mapping_dict.items()}


def data_prep(df):
    """

    :param df: dataframe
    :return: text as 'X' and labels as 'y' after mapping the text to int.
    """
    X = df['text']
    y = df['type'].map(mapping_dict)
    return X,y

def vetorize_data(X,y):
    """

    :param X: text
    :param y: label
    :return: vectorizer model, vectorized train text and vectorized test text and the labels
    """
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(X.values.astype('U'))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return tfidf, X_train, X_test, y_train, y_test

def train_model(X_train,y_train):
    """

    :param X_train: data to be trained
    :param y_train: labels to be trained
    :return: model trained on the data
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def accuracy_score(model,X_test,y_test):
    """
    :param model: trained model
    :param X_test: test data
    :param y_test: test label of the data
    :return: accuracy score of the model
    """
    return model.score(X_test, y_test)

def predict(text,model,vectorizer_model):
    """

    :param text: string to test
    :param model: trained model
    :param vectorizer_model: tfidf vectorizer model to vectorize the text
    :return: predicted target
    """
    data = [text]
    vect = vectorizer_model.transform(data).toarray()
    my_prediction = model.predict(vect)
    return inv_mapping_dict[my_prediction[0]]


def main():
    df = pd.read_csv("data/hamOrspam.csv", encoding="latin-1")
    X, y = data_prep(df)

    vectorizer_model, X_train, X_test, y_train, y_test = vetorize_data(X, y)

    model = train_model(X_train, y_train)
    print("Accuracy:", accuracy_score(model, X_test, y_test))

    test_text = "You have won 1 Lakh lottery! please click the link below to claim the money"
    print(predict(test_text, model, vectorizer_model))


if __name__ == '__main__':
    main()
