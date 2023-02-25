"""
Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
Article: https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
Github: https://github.com/nkartik94/Multi-Label-Text-Classification/blob/master/Mark_6.ipynb
"""
import re
import warnings

import numpy as np
import pandas as pd
import sys
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

if not sys.warnoptions:
    warnings.simplefilter("ignore")

data_raw = pd.read_csv("../data/toxic_comments/train.csv")

missing_values_check = data_raw.isnull().sum()
print(missing_values_check)

rowSums = data_raw.iloc[:, 2:].sum(axis=1)
clean_comments_count = (rowSums == 0).sum(axis=0)

print("Total number of comments = ", len(data_raw))
print("Number of clean comments = ", clean_comments_count)
print("Number of comments with labels =", (len(data_raw) - clean_comments_count))

categories = list(data_raw.columns.values)
categories = categories[2:]
print(categories)

counts = []
for category in categories:
    counts.append((category, data_raw[category].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number of comments'])
print(df_stats)

# 2. Data Pre-Processing
print("\n2. Data Pre-Processing")
data = data_raw.loc[np.random.choice(data_raw.index, size=2000)]
print(data.shape)

# 2.1. Cleaning Data
print("\n2.1. Cleaning Data")


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence):  # function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


data['comment_text'] = data['comment_text'].str.lower()
data['comment_text'] = data['comment_text'].apply(cleanHtml)
data['comment_text'] = data['comment_text'].apply(cleanPunc)
data['comment_text'] = data['comment_text'].apply(keepAlpha)
print(data[["comment_text"]].head())

# 2.2. Removing Stop Words
print("\n2.2. Removing Stop Words")
stop_words = set(stopwords.words('english'))
stop_words.update(
    ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also', 'across',
     'among', 'beside', 'however', 'yet', 'within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)


def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


data['comment_text'] = data['comment_text'].apply(removeStopWords)
print(data[["comment_text"]].head())

# 2.3. Stemming
print("\n2.3. Stemming")
stemmer = SnowballStemmer("english")


def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


data['comment_text'] = data['comment_text'].apply(stemming)
data.head()

data['comment_text'] = data['comment_text'].apply(removeStopWords)
print(data[["comment_text"]].head())

# 2.4. Train-Test Split
print("\n2.4. Train-Test Split")

train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)

train_text = train['comment_text']
test_text = test['comment_text']

print(train.shape)
print(test.shape)

# 2.5. TF-IDF
print("\n2.5. TF-IDF")
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1, 3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels=['id', 'comment_text'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels=['id', 'comment_text'], axis=1)

# 3. Multi-Label Classification
print("\n3. Multi-Label Classification")

# 3.1. Multiple Binary Classifications - (One Vs Rest Classifier)
print("\n3.1. Multiple Binary Classifications - (One Vs Rest Classifier)")

LogReg_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
])

for category in categories:
    print('**Processing {} comments...**'.format(category))

    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])

    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print("\n")

# 3.2. Multiple Binary Classifications - (Binary Relevance)
print("\n3.2. Multiple Binary Classifications - (Binary Relevance)")

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(x_train, y_train)

# predict
predictions = classifier.predict(x_test)

# accuracy
print("Accuracy = ", accuracy_score(y_test, predictions))
print("\n")

# 3.3. Classifier Chains
print("\n3.3. Classifier Chains")

# initialize classifier chains multi-label classifier
classifier = ClassifierChain(LogisticRegression())

# Training logistic regression model on train data
classifier.fit(x_train, y_train)

# predict
predictions = classifier.predict(x_test)

# accuracy
print("Accuracy = ", accuracy_score(y_test, predictions))
print("\n")

# 3.4. Label Powerset
print("\n3.4. Label Powerset")
# initialize label powerset multi-label classifier
classifier = LabelPowerset(LogisticRegression())

# train
classifier.fit(x_train, y_train)

# predict
predictions = classifier.predict(x_test)

# accuracy
print("Accuracy = ", accuracy_score(y_test, predictions))
print("\n")

# 3.5. Adapted Algorithm
print("\n3.5. Adapted Algorithm")

classifier_new = MLkNN(k=10)

# Note that this classifier can throw up errors when handling sparse matrices.

x_train = lil_matrix(x_train).toarray()
y_train = lil_matrix(y_train).toarray()
x_test = lil_matrix(x_test).toarray()

# train
classifier_new.fit(x_train, y_train)

# predict
predictions_new = classifier_new.predict(x_test)

# accuracy
print("Accuracy = ", accuracy_score(y_test, predictions_new))
print("\n")
