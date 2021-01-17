# Include Libraries

import pandas as pnd
from sklearn.model_selection import train_test_split


import sklearn

from pandas_ml import ConfusionMatrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfiderfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from matplotlib import pyplot as plot
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import itertools

import numpy as num


# Importing dataset using the pandas dataframe
derf = pnd.read_csv("NewsArticle.csv")

derf.shape

derf.head()

# Set index
derf = derf.set_index("Unnamed: 0")

# Print first lines of `derf`
derf.head()


# Separate the labels and set up training and test datasets

y = derf.label

# Drop the `label` column
derf.drop("label", axis=1)      #where numbering of news article is done that column is dropped in dataset

# Make training and test sets
X_train, X_test, y_train, y_test = train_test_split(derf['text'], y, test_size=0.33, random_state=53)


# Building the Count and Tfiderf Vectors


count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train)                  # Learn the vocabulary dictionary and return term-document matrix.

count_test = count_vectorizer.transform(X_test)

# Initialize the `tfiderf_vectorizer`
tfiderf_vectorizer = TfiderfVectorizer(stop_words='english', max_derf=0.7)    # This removes words which appear in more than 70% of the articles

# Fit and transform the training data
tfiderf_train = tfiderf_vectorizer.fit_transform(X_train)

# Transform the test set
tfiderf_test = tfiderf_vectorizer.transform(X_test)


count_derf = pnd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

tfiderf_derf = pnd.DataFrame(tfiderf_train.A, columns=tfiderf_vectorizer.get_feature_names())

difference = set(count_derf.columns) - set(tfiderf_derf.columns)

print(difference)

# Check whether the DataFrames are equal
print(count_derf.equals(tfiderf_derf))

print(count_derf.head())

print(tfiderf_derf.head())


#--------------------------------------------------------------
# Function to plot the confusion matrix
#--------------------------------------------------------------

def plconMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plot.cm.Blues):

    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = num.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, num.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')


#--------------------------------------------------------------
# Naive Bayes classifier for Multinomial model
#--------------------------------------------------------------

clerf = MultinomialNB()

clerf.fit(tfiderf_train, y_train)                       # Fit Naive Bayes classifier according to X, y

pred = clerf.predict(tfiderf_test)                     # Perform classification on an array of test vectors X.
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plconMatrix(cm, classes=['FAKE', 'REAL'])
print(cm)


clerf = MultinomialNB()

clerf.fit(count_train, y_train)

pred = clerf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plconMatrix(cm, classes=['FAKE', 'REAL'])
print(cm)


#  Passive Aggressive Classifier

linear_clerf = PassiveAggressiveClassifier(n_iter=50)

linear_clerf.fit(tfiderf_train, y_train)
pred = linear_clerf.predict(tfiderf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plconMatrix(cm, classes=['FAKE', 'REAL'])
print(cm)


clerf = MultinomialNB(alpha=0.1)               # Additive (Laplace/Lidstone) smoothing parameter

last_score = 0
for alpha in num.arange(0,1,.1):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(tfiderf_train, y_train)
    pred = nb_classifier.predict(tfiderf_test)
    score = metrics.accuracy_score(y_test, pred)
    if score > last_score:
        clerf = nb_classifier
    print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))


def binaryClassification(vectorizer, classifier, n=100):       # inspect the top 30 vectors for fake and real news
    """

    Identify most important features if given a vectorizer and binary classifier. Set n to the number
    of weighted features you would like to show. (Note: current implementation merely prints and does not
    return top classes.)
    """

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()                                            # Array mapping from feature integer indices to feature name
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


binaryClassification(tfiderf_vectorizer, linear_clerf, n=30)
feature_names = tfiderf_vectorizer.get_feature_names()

### Most real
sorted(zip(clerf.coef_[0], feature_names), reverse=True)[:20]

### Most fake
sorted(zip(clerf.coef_[0], feature_names))[:20]                               # clearly there are certain words which might show political intent and source in the top fake features (such as the words corporate and establishment).

tokens_with_weights = sorted(list(zip(feature_names, clerf.coef_[0])))
#print(tokens_with_weights)

# HashingVectorizer : require less memory and are faster (because they are sparse and use hashes rather than tokens)


hash_vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
hash_train = hash_vectorizer.fit_transform(X_train)
hash_test = hash_vectorizer.transform(X_test)

# Naive Bayes classifier for Multinomial model


clerf = MultinomialNB(alpha=.01)

clerf.fit(hash_train, y_train)
pred = clerf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plconMatrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# Applying Passive Aggressive Classifier

clerf = PassiveAggressiveClassifier(n_iter=50)

clerf.fit(hash_train, y_train)
pred = clerf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plconMatrix(cm, classes=['FAKE', 'REAL'])
print(cm)
