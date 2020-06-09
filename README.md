# Work in Progress 
### I'll update it soon!!

# Magic of TF-IDF
Term Frequency Inverse Document Frequency (TFIDF) can do wonders! 

While dealing with text, the issue is that the machines don't understand the raw text. So, for them to understand the text, we need to convert them into numeric form. And to do that, we can use TFIDF which is one of the text vectorization methods that transform the text into vectors. These vectors can then be plotted to see how each point is placed in a vector space. Once we have the vectors, we can use any Machine Learning or Deep Learning Algorithms!<br>
<br>```TFIDF was introduced to improve the result of Bag of words (BoW). By the way, did you know that Term Frequency - Inverse Document Frequency was introduced in a 1972 paper by Karen Spärck Jones under the name "term specificity"? 😲 ```<br>
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

<b>Now let's get started!!! </b><br>

# Term Frequency Inverse Document Frequency (TFIDF)
<hr> </hr>
<b>Term Frequency Inverse Document Frequency (TFIDF)</b> is a statistical measure that reflects how important a word is to a document. TF-IDF is mostly used for document search and information retrieval through scoring that gives the importance of the word in a document. The higher the TFIDF score, the rarer the term, and vise versa. <br>
TF-IDF for a word in a document is calculated by multiplying two different metrics: <u>term frequency</u>, and <u>inverse document frequency</u>.<br>
```TFIDF = TF * IDF```<br>
<i>where,</i><br>
TF(term) = Number of times the term appears in document / total number of terms in the document<br>
IDF(term) = log(total number of documents / Number of documents with term in it)
