from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def get_vectors(str1,str2):
    """

    :param str1: string1
    :param str2: string2
    :return: vector of string1 and vector os string2
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit_transform([str1,str2])

    vector1 = tfidf.transform([str1]).toarray()
    vector2 = tfidf.transform([str2]).toarray()

    return vector1,vector2


def main():
    str1 = "This is a foo bar sentence ."
    str2 = "This sentence is similar to a foo bar sentence ."

    vector1, vector2 = get_vectors(str1,str2)

    cosine = cosine_similarity(vector1, vector2)
    print("Cosine:", cosine[0][0])

if __name__ == '__main__':
    main()