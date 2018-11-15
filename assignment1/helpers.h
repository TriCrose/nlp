#pragma once
#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <map>
#include <filesystem>
#include <functional>

enum class Sentiment : bool {
    POSITIVE = true,
    NEGATIVE = false
};

std::vector<std::vector<std::filesystem::path>> GetFolds(const std::filesystem::path& data_dir, int no_folds);

void ProcessTokens(const std::filesystem::path& path, std::function<void(const std::string&)> callback);

void GetTotals(const std::vector<std::vector<std::filesystem::path>>& folds,
               int test_set_index,
               std::map<std::string, int>* word_freq,
               std::map<std::string, std::map<Sentiment, int>>* word_sent_freq,
               std::map<Sentiment, int>* words_per_sentiment);

float NaiveBayes(const std::vector<std::filesystem::path>& test_set,
                 const std::map<std::string, std::map<Sentiment, double>>& log_probs);

#endif // HELPERS_H