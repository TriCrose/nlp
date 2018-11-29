import os
import re

from random import shuffle
from sklearn.svm import SVC
from stemmer import PorterStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

stemmer = PorterStemmer()

def tokenize(path, stemming):
    with open(path, encoding="utf-8") as f:
        text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", f.read().lower().replace("<br />", " "))
        return [stemmer.stem(t, 0, len(t)-1) for t in text.split()] if stemming else text.split()

def classify(folds, stemming):
    print("Building", "stemmed" if stemming else "unstemmed", "doc2vec model...", end=" ", flush=True)
    data_path = "data/large/"
    documents = []

    for i, path in enumerate([data_path + f for f in os.listdir(data_path)]):
        documents.append(TaggedDocument(tokenize(path, stemming), [i]))

    model = Doc2Vec(documents, min_count=1, window=10, vector_size=100, hs=1, workers=8)
    print("done", flush=True)

    print("Accuracy:", end=" ", flush=True)
    paths = [f[0] for fold in folds for f in fold]
    X = [model.infer_vector(doc) for doc in [tokenize(p, stemming) for p in paths]]
    accuracies = []

    for i in range(len(folds)):
        classifier = SVC(gamma="auto", kernel="linear")

        start_index = 0
        for j in range(i):
            start_index += len(folds[j])
        end_index = start_index + len(folds[i])

        test_set = X[start_index:end_index]
        training_set = X[:start_index] + X[end_index:]
        classifier.fit(training_set, [f[1] for fold in (folds[:i] + folds[i+1:]) for f in fold])

        correct_predictions = 0
        results = classifier.predict(test_set)
        for j in range(len(results)):
            correct_predictions += int(results[j] == folds[i][j][1])

        accuracies.append(100 * correct_predictions/len(results))

    print(sum(accuracies)/len(accuracies), end="\n\n", flush=True)

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
    classify(folds, True)
    classify(folds, False)

perform_tests(3)