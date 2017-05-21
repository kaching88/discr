#ifndef MDLP_HPP
#define MDLP_HPP

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <forward_list>

namespace Utils {
    template <typename T>
    std::vector<size_t> order(const std::vector<T> &vec);

    template <typename T>
    std::vector<T> apply_index(std::vector<T> &vec, const std::vector<size_t> &indices);
}

class MDLP {
    static std::unordered_map<size_t, size_t> count_frequencies(std::vector<size_t>::iterator start,
                                                                std::vector<size_t>::iterator end);

    static double entropy(std::vector<size_t>::iterator start,
                          std::vector<size_t>::iterator end,
                          std::unordered_map<size_t, size_t> frequencies);

    static std::tuple<bool, size_t> check_duplicate(std::vector<double> &x,
                                                    std::vector<size_t> &y,
                                                    size_t start);

    static std::tuple<size_t, double, double, double, std::unordered_map<size_t, size_t>, std::unordered_map<size_t, size_t>> find_best_split(std::vector<double> &x,
                                                                                                                                              std::vector<size_t> &y,
                                                                                                                                              size_t start,
                                                                                                                                              size_t end);

    static bool mdlpc_criterion(const std::tuple<size_t, double, double, double, std::unordered_map<size_t, size_t>, std::unordered_map<size_t, size_t>> &bests,
                                std::vector<size_t>::iterator start,
                                std::vector<size_t>::iterator end);

    static std::forward_list<size_t> part(std::vector<double> &x,
                                          std::vector<size_t> &y,
                                          size_t start,
                                          size_t end);

    MDLP();

public:
    static std::vector<double> mdlp_implementation(std::vector<double> x, std::vector<size_t> y);
};

#endif // MDLP_HPP
