#include <iostream>
#include <cmath>

#include "helpers.h"

int main() {
    constexpr auto NUM_FOLDS = 3;
    auto folds = GetFolds("../../data", NUM_FOLDS);

    for (auto i = 0; i < NUM_FOLDS; i++) {
        auto word_freq = std::map<std::string, int>{};
        auto word_sent_freq = std::map<std::string, std::map<Sentiment, int>>{};
        auto words_per_sentiment = std::map<Sentiment, int>{};
        GetTotals(folds, i, &word_freq, &word_sent_freq, &words_per_sentiment);

        std::cout << "Unique words: " << word_freq.size() << "\n"
                  << "Words in positive reviews: " << words_per_sentiment[Sentiment::POSITIVE] << "\n"
                  << "Words in negative reviews: " << words_per_sentiment[Sentiment::NEGATIVE] << "\n";

        auto unsmoothed_log_probs = std::map<std::string, std::map<Sentiment, double>>{};
        auto smoothed_log_probs = std::map<std::string, std::map<Sentiment, double>>{};

        std::cout << "Calculating unsmoothed log probabilities...";
        for (auto& entry : word_sent_freq) {
            unsmoothed_log_probs[entry.first][Sentiment::POSITIVE] = std::log(
                static_cast<double>(entry.second[Sentiment::POSITIVE])
                / static_cast<double>(words_per_sentiment[Sentiment::POSITIVE])
            );

            unsmoothed_log_probs[entry.first][Sentiment::NEGATIVE] = std::log(
                static_cast<double>(entry.second[Sentiment::NEGATIVE])
                / static_cast<double>(words_per_sentiment[Sentiment::NEGATIVE])
            );
        }
        std::cout << " done\n";

        std::cout << "Calculating smoothed log probabilities...";
        for (auto& entry : word_sent_freq) {
            smoothed_log_probs[entry.first][Sentiment::POSITIVE] = std::log(
                static_cast<double>(entry.second[Sentiment::POSITIVE] + 1)
                / static_cast<double>(words_per_sentiment[Sentiment::POSITIVE] + static_cast<int>(word_freq.size()))
            );

            smoothed_log_probs[entry.first][Sentiment::NEGATIVE] = std::log(
                static_cast<double>(entry.second[Sentiment::NEGATIVE] + 1)
                / static_cast<double>(words_per_sentiment[Sentiment::NEGATIVE] + static_cast<int>(word_freq.size()))
            );
        }
        std::cout << " done\n";

        std::cout << "Running unsmoothed Naive Bayes: "
                  << 100.0f * NaiveBayes(folds[i], unsmoothed_log_probs) << "% accuracy\n";
        std::cout << "Running smoothed Naive Bayes: "
                  << 100.0f * NaiveBayes(folds[i], smoothed_log_probs) << "% accuracy\n";

        std::cout << "\n";
    }

    system("PAUSE");
    return 0;
}