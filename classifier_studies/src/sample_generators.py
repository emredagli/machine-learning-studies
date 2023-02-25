""" Samples generators
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
"""

from sklearn.datasets import make_multilabel_classification, make_blobs, make_classification

""" make_blobs
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
Generate isotropic Gaussian blobs for clustering.
"""
X, y = make_blobs(
    n_samples=10,
    centers=3,
    n_features=2,
    random_state=0)
"""
X[0:5]
array([[ 1.12031365,  5.75806083],
       [ 1.7373078 ,  4.42546234],
       [ 2.36833522,  0.04356792],
       [ 0.87305123,  4.71438583],
       [-0.66246781,  2.17571724]])
y[0:5]
array([0, 0, 1, 0, 2])       
"""

X, y = make_classification(
    n_samples=1000,  # 1000 observations
    n_features=5,  # 5 total features
    n_informative=3,  # 3 'useful' features
    n_classes=2,  # binary target/label
    random_state=0  # if you want the same results as mine
)
"""
X[0:5]
array([[ 1.20519726,  0.44258782,  0.91819574,  2.06906552,  1.99604691],
       [ 1.46305482,  0.44920227,  1.10737169,  1.54372757,  0.69062924],
       [-0.22465199,  0.20949367, -0.12463027,  1.10623046,  2.23373344],
       [ 2.17197093, -3.46329621,  0.74581999, -0.40963294, -2.21130493],
       [ 2.21242095,  0.20313378,  1.58235924,  1.15371571, -0.91704557]])
y[0:5]
array([1, 1, 0, 0, 1])
"""

X, y, p_c, p_w_c = make_multilabel_classification(
    n_samples=1000,
    n_features=100,
    n_classes=5,  # The total number of distinct labels
    n_labels=2,  # The average number of labels per instance.
    length=10000,  # The sum of the features (number of words if documents)
    # is drawn from a Poisson distribution with this expected value.
    allow_unlabeled=False,  # If ``True``, some instances might not belong to any class.
    sparse=False,  # default=False, If ``True``, return a sparse feature matrix.
    return_distributions=True,
    random_state=0
)

print(type(X))

"""
X[0:5]
array([[15., 13., 16.,  5.],
       [ 8., 12., 20., 11.],
       [14., 12., 15.,  8.],
       [12.,  9., 12., 14.],
       [19., 16., 14., 19.]])
y[0:5]       
array([[0, 1],
       [1, 1],
       [0, 1],
       [1, 1],
       [1, 1]])
"""

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
