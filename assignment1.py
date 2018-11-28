import os

from stemmer import PorterStemmer
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

def naive_bayes(folds, smoothed, ngrams, stemming, binary):
    p = PorterStemmer()

    vectorizer = CountVectorizer(input="filename", \
                                 ngram_range=ngrams, \
                                 tokenizer=(lambda d: [(p.stem(t, 0, len(t)-1) if stemming else t) for t in d.split()]), \
                                 binary=binary, \
                                 min_df=4, max_df=1.0)

    X = vectorizer.fit_transform([f for fold in folds for f in fold])

    accuracies = []
    for i in range(len(folds)):
        classifier = MultinomialNB(alpha=(1.0 if smoothed else 0))

        for j in range(len(folds)):
            if i == j:
                continue

            start_index = 0
            for k in range(j):
                start_index += len(folds[k])
            fold_length = len(folds[j])
            end_index = start_index + fold_length

            classifier.partial_fit(X[start_index:end_index], [True] * (fold_length // 2) + \
                                                             [False] * (fold_length // 2), \
                                                             [True, False])

        start_index = 0
        for j in range(i):
            start_index += len(folds[j])
        end_index = start_index + len(folds[i])

        correct_predictions = 0
        results = classifier.predict(X[start_index:end_index])
        for j in range(len(results)):
            correct_predictions += int(results[j] == (j < len(folds[i]) // 2))

        accuracies.append(100 * correct_predictions/len(results))

    print("smoothed" if smoothed else "unsmoothed", \
          "stemmed" if stemming else "unstemmed", \
          "presence" if binary else "frequency", \
          "unigrams" if ngrams == (1, 1) else \
          ("bigrams" if ngrams == (2, 2) else \
          ("uni + bi" if ngrams == (1, 2) else "unknown")), \
          "accuracy:", sum(accuracies)/len(accuracies))

def perform_tests(num_folds):
    pos_path = "data/POS/"
    neg_path = "data/NEG/"
    pos_files = [pos_path + p for p in sorted(os.listdir(pos_path))]
    neg_files = [neg_path + p for p in sorted(os.listdir(neg_path))]

    folds = [[] for i in range(num_folds)]
    for i in range(len(pos_files)):
        folds[i % num_folds].append(pos_files[i])
    for i in range(len(neg_files)):
        folds[i % num_folds].append(neg_files[i])

    print("\nUsing", num_folds, "fold cross-validation.\n")

    print("Naive Bayes\n***********")
    for smoothed in [True, False]:
        for stemming in [True, False]:
            for binary in [True, False]:
                for ngrams in [(1, 1), (2, 2), (1, 2)]:
                    naive_bayes(folds, smoothed, ngrams, stemming, binary)

perform_tests(3)