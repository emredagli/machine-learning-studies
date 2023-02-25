from sklearn.datasets import make_multilabel_classification
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns


X, Y = make_multilabel_classification(
    n_classes=2, n_labels=1, allow_unlabeled=True, random_state=1
)

"""
FROM: https://scikit-learn.org/stable/datasets/sample_generators.html
make_multilabel_classification generates random samples with multiple labels, reflecting a bag of words drawn from a mixture of topics. 
The number of topics for each document is drawn from a Poisson distribution, and the topics themselves are drawn from a fixed random distribution. 
Similarly, the number of words is drawn from Poisson, with words drawn from a multinomial, where each topic defines a probability distribution over words. 
Simplifications with respect to true bag-of-words mixtures include:
* Per-topic word distributions are independently drawn, where in reality all would be affected by a sparse base distribution, and would be correlated.
* For a document generated from multiple topics, all topics are weighted equally in generating its bag of words.
* Documents without labels words at random, rather than from a base distribution.
"""