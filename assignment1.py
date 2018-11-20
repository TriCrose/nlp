import os
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB

# Independent variable columns:
# *****************************
# Features: unigrams, bigrams, unigrams + bigrams
# Stemming: yes, no

# Dependent variable columns:
# ***************************
# Unsmoothed Naive Bayes
# Smoothed Naive Bayes
# SVM

# 1. Load all documents as maps from string to int (word occurrences)
# 2. Produce a list of every word that occurs in the dataset
# 3. Create a sparse 2000 x (no_of_features) matrix to store all the documents
# 4. Train
# 5. Classify

# X = lil_matrix((1000, 60000))
# y = np.random.randint(5, size=1000)

# classifier = MultinomialNB()
# classifier.fit(X, y)

# print(classifier.predict(lil_matrix((1, 60000))))

# TODO: add support for unigrams/bigrams
def get_doc_vector(file, word_freqs):
    doc = {}

    for token in file.readlines():
        if not token in word_freqs:
            word_freqs[token] = 1
        else:
            word_freqs[token] += 1

        if not token in doc:
            doc[token] = 1
        else:
            doc[token] += 1

    return doc

def load_data(num_folds):
    pos_path = "data/POS/"
    neg_path = "data/NEG/"

    pos_files = os.listdir(pos_path)
    neg_files = os.listdir(neg_path)
    pos_files.sort()
    neg_files.sort()

    folds = [[] for i in range(num_folds)]
    word_freqs = {}

    for i in range(1000):
        with open(pos_path + pos_files[i], encoding="utf8") as f:
            folds[i % num_folds].append(get_doc_vector(f, word_freqs))
        with open(neg_path + neg_files[i], encoding="utf8") as f:
            folds[i % num_folds].append(get_doc_vector(f, word_freqs))

    return folds, word_freqs

options = {
    "folds": 3,
    "min_freq_cutoff": 4,
    "unigrams": True,
    "bigrams": False
}