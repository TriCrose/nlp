import os
from collections import Counter
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from stemmer import PorterStemmer

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
def get_doc_vector(file, word_freqs, stemming):
    stemmer = PorterStemmer()
    doc = Counter()
    for token in file.readlines():
        token = token.strip().lower()
        if not token:
            continue
        if stemming:
            token = stemmer.stem(token, 0, len(token) - 1)
        word_freqs[token] += 1
        doc[token] += 1
    return doc

def load_documents(num_folds, stemming):
    pos_path = "data/POS/"
    neg_path = "data/NEG/"

    pos_files = os.listdir(pos_path)
    neg_files = os.listdir(neg_path)
    pos_files.sort()
    neg_files.sort()

    folds = [[] for i in range(num_folds)]
    word_freqs = Counter()

    def load_files(files, base_path):
        for i in range(len(files)):
            with open(base_path + files[i], encoding="utf8") as f:
                folds[i % num_folds].append(get_doc_vector(f, word_freqs, stemming))

    load_files(pos_files, pos_path)
    load_files(neg_files, neg_path)

    return folds, word_freqs

NUM_FOLDS = 3
MIN_FREQ_CUTOFF = 4
STEMMING = False

folds, word_freqs = load_documents(NUM_FOLDS, STEMMING)
# TODO