# Necessary imports
import pandas as pd
import numpy as np
from numpy import genfromtxt
# from pymfe.mfe import MFE
import csv
import arff

data = arff.load(open('../data/multi-label/ENRON-F.arff', 'r')) # ['data']
X = [i[:4] for i in data]
y = [i[-1] for i in data]

print("X shape --> ", len(X))
print("y shape --> ", len(y))
print("classes --> ", np.unique(y))
print("X dtypes --> ", type(X))
print("y dtypes --> ", type(y))


# X, Y = make_multilabel_classification(
#     n_classes=2, n_labels=1, allow_unlabeled=True, random_state=1
# )