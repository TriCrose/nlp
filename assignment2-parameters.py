import os
import re

from itertools import product
from stemmer import PorterStemmer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

stemmer = PorterStemmer()

def tokenize(path, stemming):
    with open(path, encoding="utf-8") as f:
        text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", f.read().lower().replace("<br />", " "))
        return [stemmer.stem(t, 0, len(t)-1) for t in text.split()] if stemming else text.split()

def classify(input_vectors, training_set, validation_set, file_path):
    classifier = SVC(gamma="auto", kernel="linear")
    classifier.fit(input_vectors[:len(training_set)], [f[1] for f in training_set])
    results = classifier.predict(input_vectors[len(training_set):])

    correct_predictions = 0
    with open(file_path, "w") as prediction_file:
        for i in range(len(results)):
            path = validation_set[i][0]
            true_sent = validation_set[i][1]
            pred_sent = results[i]

            correct_predictions += int(true_sent == pred_sent)
            prediction_file.write(path + ": " + str(true_sent) + ", predicted: " + str(pred_sent) + "\n")

    return correct_predictions/len(results)

# optimal config is uni + bi, stemmed, presence
def svm_bow(training_set, validation_set):
    optimal_config = (0,)

    for ngrams, stemming, binary in product(((1, 1), (2, 2), (1, 2)), (True, False), (True, False)):
        filename = ("uni" if ngrams == (1, 1) else ("bi" if ngrams == (2, 2) else "unibi")) + "_" + \
                   ("stemmed" if stemming else "unstemmed") + "_" + \
                   ("presence" if binary else "freq") + ".txt"
        print("Testing config " + filename + "...", end=" ", flush=True)

        vectorizer = CountVectorizer( \
            input="filename", \
            ngram_range=ngrams, \
            tokenizer=lambda d: [stemmer.stem(t, 0, len(t)-1) for t in d.split()] if stemming else d.split(), \
            binary=binary, \
            min_df=4, max_df=1.0 \
        )

        input_vectors = vectorizer.fit_transform([f[0] for f in training_set + validation_set])
        accuracy = classify(input_vectors, training_set, validation_set, "predictions_bow/" + filename)

        if accuracy > optimal_config[0]:
            optimal_config = (accuracy, ngrams, stemming, binary)
        print("accuracy", str(100 * accuracy) + "%", flush=True)

    return optimal_config[1:]

# optimal config is unstemmed, 5 window, 20 epochs
def svm_d2v(training_set, validation_set):
    optimal_config = (0,)

    data_path = "data/large/"
    stemmed_docs = []
    unstemmed_docs = []

    print("Loading and tokenizing large dataset...", end=" ", flush=True)
    for i, path in enumerate([data_path + f for f in os.listdir(data_path)]):
        stemmed_docs.append(TaggedDocument(tokenize(path, True), [i]))
        unstemmed_docs.append(TaggedDocument(tokenize(path, False), [i]))
    print("done\n")

    for stemming, window, epochs in product((True, False), (2, 5, 10), (10, 20)):
        filename = ("stemmed" if stemming else "unstemmed") + "_" + \
                   str(window) + "window_" + \
                   str(epochs) + "epoch.txt"

        print("Building doc2vec model for " + filename + "...", end=" ", flush=True)
        model = Doc2Vec( \
            stemmed_docs if stemming else unstemmed_docs, \
            epochs=epochs, window=window, \
            min_count=1, vector_size=100, hs=1 \
        )
        print("done, classifying...", end=" ", flush=True)

        paths = [f[0] for f in training_set + validation_set]
        input_vectors = [model.infer_vector(doc) for doc in [tokenize(p, stemming) for p in paths]]
        accuracy = classify(input_vectors, training_set, validation_set, "predictions_d2v/" + filename)

        if accuracy > optimal_config[0]:
            optimal_config = (accuracy, stemming, window, epochs)
        print("accuracy", str(100 * accuracy) + "%", flush=True)

    return optimal_config[1:]

pos_path = "data/POS/"
neg_path = "data/NEG/"
pos_files = [pos_path + p for p in sorted(os.listdir(pos_path))]
neg_files = [neg_path + p for p in sorted(os.listdir(neg_path))]

training_set = []
validation_set = []

for i in range(len(pos_files)):
    (training_set if i % 10 else validation_set).append((pos_files[i], True))
for i in range(len(neg_files)):
    (training_set if i % 10 else validation_set).append((neg_files[i], False))

print(svm_bow(training_set, validation_set))
print(svm_d2v(training_set, validation_set))