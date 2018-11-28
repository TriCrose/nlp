import os

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack

from stemmer import PorterStemmer

def classify(folds, nb_or_svm, ngrams, stemming, binary):
    p = PorterStemmer()

    vectorizer = CountVectorizer(input="filename", \
                                 ngram_range=ngrams, \
                                 tokenizer=(lambda d: [(p.stem(t, 0, len(t)-1) if stemming else t) for t in d.split()]), \
                                 binary=binary, \
                                 min_df=4, max_df=1.0)

    X = vectorizer.fit_transform([f[0] for fold in folds for f in fold])

    accuracies = []
    for i in range(len(folds)):
        classifier = SVC(gamma="auto", kernel="linear") if nb_or_svm[0] == "svm" \
                else MultinomialNB(alpha=(1.0 if nb_or_svm[1] else 1.0e-10))

        start_index = 0
        for j in range(i):
            start_index += len(folds[j])
        end_index = start_index + len(folds[i])

        test_set = X[start_index:end_index]
        training_set = vstack([X[:start_index], X[end_index:]])
        classifier.fit(training_set, [f[1] for fold in (folds[:i] + folds[i+1:]) for f in fold])

        correct_predictions = 0
        results = classifier.predict(test_set)
        for j in range(len(results)):
            correct_predictions += int(results[j] == folds[i][j][1])

        accuracies.append(100 * correct_predictions/len(results))

    if nb_or_svm[0] != "svm":
        print("smoothed" if nb_or_svm[1] else "unsmoothed", end=" ")

    print("stemmed" if stemming else "unstemmed", \
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
        folds[i % num_folds].append((pos_files[i], True))
    for i in range(len(neg_files)):
        folds[i % num_folds].append((neg_files[i], False))

    print("\nUsing", num_folds, "fold cross-validation.\n")

    print("SVM\n***")
    for stemming in [True, False]:
        for binary in [True, False]:
            for ngrams in [(1, 1), (2, 2), (1, 2)]:
                classify(folds, ("svm",), ngrams, stemming, binary)
    print()

    print("Naive Bayes\n***********")
    for smoothed in [True, False]:
        for stemming in [True, False]:
            for binary in [True, False]:
                for ngrams in [(1, 1), (2, 2), (1, 2)]:
                    classify(folds, ("nb", smoothed), ngrams, stemming, binary)

perform_tests(3)