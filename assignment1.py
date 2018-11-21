import os

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Independent variable columns:
# *****************************
# Features: unigrams, bigrams, unigrams + bigrams
# Stemming?
# Frequency vs Presence?

# Dependent variable columns:
# ***************************
# Unsmoothed Naive Bayes
# Smoothed Naive Bayes
# SVM

def run_naive_bayes(num_folds, smoothed, ngrams, stemming, binary):
    pos_path = "data/POS/"
    neg_path = "data/NEG/"
    files = [pos_path + p for p in sorted(os.listdir(pos_path))] + \
            [neg_path + p for p in sorted(os.listdir(neg_path))]
    print(files)

    def tokenize(document):
        return document
    vectorizer = CountVectorizer(input="filename", ngram_range=ngrams)

run_naive_bayes(3, True, (1, 1), False, False)

"""
# TODO: add support for unigrams/bigrams
def get_doc_vector(file, word_freqs, stemming):
    stemmer = PorterStemmer()
    document = Counter()
    for token in file.readlines():
        token = token.strip().lower()
        if not token:
            continue
        if stemming:
            token = stemmer.stem(token, 0, len(token) - 1)
        word_freqs[token] += 1
        document[token] += 1
    return document

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
classifier = MultinomialNB()

word_indices = {}
ctr = 0
for word in word_freqs:
    word_indices[word] = ctr
    ctr += 1

for i in range(len(folds)):
    for j in range(len(folds)):
        if i == j:
            continue
        training_vectors = sparse.dok_matrix((len(folds[j]), len(word_freqs)), dtype=np.int32)
        for doc_index in range(len(folds[j])):
            print("Document number: ", doc_index)
            for word, freq in folds[j][doc_index].items():
                word_index = word_indices[word]
                training_vectors[doc_index, word_index] = freq
        fold_size_half = len(folds[j]) // 2
        target_values = [True] * fold_size_half + [False] * fold_size_half
        classifier.partial_fit(training_vectors, target_values, classes=[True, False])
    # TODO: predict on folds[i]
"""