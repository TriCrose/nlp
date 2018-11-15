#include <iostream>
#include <sstream>
#include <fstream>

#include "helpers.h"
#include "stemmer.h"

namespace fs = std::filesystem;

std::vector<std::vector<fs::path>> GetFolds(const std::filesystem::path& data_dir, int no_folds) {
    auto folds = std::vector<std::vector<fs::path>>{};
    for (auto i = 0; i < no_folds; i++) folds.push_back({});

    auto add_paths = [&data_dir, no_folds, &folds](const std::string& dir) {
        auto paths = std::vector<fs::path>{};
        for (auto& file : fs::directory_iterator{data_dir / dir}) paths.push_back(file.path());
        std::sort(paths.begin(), paths.end());
        for (auto i = size_t{0}; i < paths.size(); i++) folds[i % no_folds].push_back(paths[i]);
    };

    add_paths("POS");
    add_paths("NEG");

    return folds;
}

void ProcessTokens(const fs::path& path, std::function<void(const std::string&)> callback) {
    auto tokens = std::vector<std::string>{};
    auto file = std::ifstream{path};
    for (auto word = std::string{}; std::getline(file, word);) {
        auto new_size = static_cast<size_t>(stem(word.data(), 0, word.size() - 1) + 1);
        word.resize(new_size);
        callback(word);
    }
}

void GetTotals(const std::vector<std::vector<fs::path>>& folds,
    int test_set_index,
    std::map<std::string, int>* word_freq,
    std::map<std::string, std::map<Sentiment, int>>* word_sent_freq,
    std::map<Sentiment, int>* words_per_sentiment)
{
    if (!word_freq || !word_sent_freq || !words_per_sentiment) {
        std::cout << "Null pointer(s) passed to GetTotals()";
        return;
    }

    word_freq->clear();
    word_sent_freq->clear();
    words_per_sentiment->clear();

    auto total_files = size_t{};
    auto current_file = size_t{0};
    for (auto i = size_t{0}; i < folds.size(); i++) if (i != test_set_index) total_files += folds[i].size();

    for (auto i = size_t{0}; i < folds.size(); i++) if (i != test_set_index) {
        for (auto j = size_t{0}; j < folds[i].size(); j++) {
            current_file++;
            std::cout << "\rProgress: " << current_file << "/" << total_files;

            auto sent = static_cast<Sentiment>(j < folds[i].size()/2);
            ProcessTokens(folds[i][j], [word_freq, word_sent_freq, words_per_sentiment, sent](const std::string& word) {
                (*word_freq)[word]++;
                (*word_sent_freq)[word][sent]++;
                (*words_per_sentiment)[sent]++;
            });
        }
    }

    std::cout << "\n";
}

float NaiveBayes(const std::vector<fs::path>& test_set,
                 const std::map<std::string, std::map<Sentiment, double>>& log_probs)
{
    auto correct_predictions = 0;

    for (auto i = size_t{0}; i < test_set.size(); i++) {
        auto actual_sentiment = static_cast<Sentiment>(i < test_set.size()/2);

        auto sum_positive = 0.0;
        auto sum_negative = 0.0;

        ProcessTokens(test_set[i], [&log_probs, &sum_positive, &sum_negative](const std::string& word) {
            if (log_probs.count(word) == 0) return;
            sum_positive += log_probs.at(word).at(Sentiment::POSITIVE);
            sum_negative += log_probs.at(word).at(Sentiment::NEGATIVE);
        });

        auto predicted_sentiment = static_cast<Sentiment>(sum_positive > sum_negative);
        correct_predictions += predicted_sentiment == actual_sentiment;
    }

    return static_cast<float>(correct_predictions)/static_cast<float>(test_set.size());
}