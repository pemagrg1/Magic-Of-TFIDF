# Magic of TF-IDF
Term Frequency Inverse Document Frequency (TFIDF) can do wonders! 

TFIDF was introduced to improve the result of Bag of words (BoW). By the way, did you know that Term Frequency - Inverse Document Frequency was introduced in a 1972 paper by Karen Spärck Jones under the name "term specificity"? 😲 <br>
coming back to the present scenario, before starting with TFIDF, let me explain BoW in brief.<br>
## Bag of Words (BoW)
A bag-of-words is a representation of text that describes the occurrence of words within a document. It's called a bag of words because it contains all the words of a document where the order and structure of the word in the document are unknown. Confusing? in simple words, it's like we have an empty bag, and we have a vocabulary of the document. And we put the words into the bad one by one. What do we get? a bag full of words. 😲<br>
![BOW image](https://cdn-images-1.medium.com/max/800/0*KwLaTHYlVY6tLASn.png)<br>
Source: https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/<br>
To make the bag of words model, [Note: taking examples from Gentle introduction to the Bag of words] <br>
1. <b>collect the data</b><br>
```
[It was the best of times,
it was the worst of times,
it was the age of wisdom,
it was the age of foolishness]
```
2. <b>Make a vocabulary of the data</b><br>
```["it", "was", "the", "best", "of", "times", "worst", "age", "wisdom", "foolishness"]```
3. <b>Create a vector</b><br>
```
"it was the worst of times" = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
"it was the age of wisdom" = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0]
"it was the age of foolishness" = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1]
```
4. <b>Score</b> the words using either count method or frequency method such as TFIDF. Which we'll be going through in this article.

<b>Now let's get started!!! </b><br><br>
<b> NOTEBOOK TO SEE THE EXECUTION:</b> https://github.com/pemagrg1/Magic-Of-TFIDF/blob/master/notebooks/TF-IDF%20from%20Scratch.ipynb

# Term Frequency Inverse Document Frequency (TFIDF)
<b>Term Frequency Inverse Document Frequency (TFIDF)</b> is a statistical measure that reflects how important a word is to a document. TF-IDF is mostly used for document search and information retrieval through scoring that gives the importance of the word in a document. The higher the TFIDF score, the rarer the term, and vise versa. <br>
TF-IDF for a word in a document is calculated by multiplying two different metrics: <u>term frequency</u>, and <u>inverse document frequency</u>.<br>
```TFIDF = TF * IDF```<br>
<i>where,</i><br>
TF(term) = Number of times the term appears in document / total number of terms in the document<br>
IDF(term) = log(total number of documents / Number of documents with term in it)

### TFIDF Applications
- Information Retrieval
- Text mining
- User Modeling
- Keyword Extraction
- Search Engine

### Term Frequency 
Term frequency(TF) is the count of a word in a document. There are several ways of calculating this frequency, with the simplest being a raw count of instances a word appears in a document.

### Inverse Document Frequency
The inverse document frequency(idf) tells us how common or rare a word is in the entire document set. The metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm. If a term spreads frequently along with other documents it can be said that it is not a relevant word such as the stop words like "the", "is", "are" etc.
<br><br>
<b>NOTE:</b> The intuition for this measure is: If a word appears frequently in a document, then it should be important and we should give that word a high score. But if a word appears in too many other documents, it's probably not a unique identifier, therefore we should assign a lower score to that word
<br><br>
#### REFERENCES:
1. https://www.kdnuggets.com/2018/08/wtf-tf-idf.html
2. https://en.wikipedia.org/wiki/Tf%E2%80%93idf
3. http://www.tfidf.com/
4. https://monkeylearn.com/blog/what-is-tf-idf/
5. https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
6. https://www.coursera.org/learn/audio-signal-processing/lecture/4QZav/dft
7. https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
8. https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
9. https://machinelearningmastery.com/gentle-introduction-bag-words-model/#:~:text=A%20bag%2Dof%2Dwords%20is,the%20presence%20of%20known%20words.

#### Additional Medium Resources For Implementations
- A Basic NLP Tutorial for News Multiclass Categorization
- Finding The Most Important Sentences Using NLP & TF-IDF
- Summarize Documents using Tf-Idf
- Document Classification
- Content Based Recommender
- Twitter sentiment analysis
- Finding Similar Quora Questions with BOW, TFIDF and Xgboost
