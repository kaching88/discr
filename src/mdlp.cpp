//#include <Rcpp.h>
#include "../include/mdlp.hpp"

template <typename T>
std::vector<size_t> Utils::order(const std::vector<T> &vec) {
  std::vector<size_t> ids(vec.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(),
            [&vec](size_t i1, size_t i2) {return vec[i1] < vec[i2];});
  return ids;
}

template <typename T>
std::vector<T> Utils::apply_index(std::vector<T> &vec, const std::vector<size_t> &indices) {
    std::vector<T>result;
    result.reserve(vec.size());
    for (auto &i : indices) {
        result.push_back(vec[i]);
    }
    return result;
}

std::unordered_map<size_t, size_t> MDLP::count_frequencies(std::vector<size_t>::iterator start,
                                                     std::vector<size_t>::iterator end) {
    std::unordered_map<size_t, size_t> frequencies;
    for (auto iter = start; iter != end; iter++) {
        size_t k = *iter;
        frequencies[k] += 1;
    }
    return frequencies;
}

double MDLP::entropy(std::vector<size_t>::iterator start,
               std::vector<size_t>::iterator end,
               std::unordered_map<size_t, size_t> frequencies) {
    double n = (double) std::distance(start, end);
    double ent = 0;
    for (auto freq : frequencies) {
        double prob = freq.second / n;
        ent += prob * log(prob);
    }
    return -ent;
}

std::tuple<bool, size_t> MDLP::check_duplicate(std::vector<double> &x, std::vector<size_t> &y, size_t start) {
    size_t i = start;
    bool same_category = true;
    while (i + 1 < x.size() && x[i] == x[i + 1]) {
        if (same_category && y[i] != y[i + 1]) {
            same_category = false;
        }
        i++;
    }
    return std::make_tuple(same_category, i + 1);
}

std::tuple<size_t, double, double, double, std::unordered_map<size_t, size_t>, std::unordered_map<size_t, size_t>> MDLP::find_best_split(std::vector<double> &x,
                                                                           std::vector<size_t> &y,
                                                                           size_t start,
                                                                           size_t end) {
    double best_cie = std::numeric_limits<double>::max();
    double best_lower_entropy = std::numeric_limits<double>::max();
    double best_upper_entropy = std::numeric_limits<double>::max();
    std::unordered_map<size_t, size_t> best_freq_lower;
    std::unordered_map<size_t, size_t> best_freq_upper;
    double n = (double) end - start;
    size_t best_index = 0;
    for (size_t i = start; i < end - 1; i++) {
        if (x[i] != x[i + 1]) {
            auto lower_begin = y.begin() + start;
            auto lower_end = y.begin() + i + 1;
            auto upper_begin = lower_end;
            auto upper_end = y.begin() + end;
            size_t n_lower_range = i - start + 1;
            size_t n_upper_range = end - i - 1;
            std::unordered_map<size_t, size_t> freq_lower = count_frequencies(lower_begin, lower_end);
            std::unordered_map<size_t, size_t> freq_upper = count_frequencies(upper_begin, upper_end);
            size_t k1 = freq_lower.size();
            size_t k2 = freq_upper.size();
            double lower_entropy = entropy(lower_begin, lower_end, freq_lower);
            double upper_entropy = entropy(upper_begin, upper_end, freq_upper);
            double lower_cie = (n_lower_range / n) * lower_entropy;
            double upper_cie = (n_upper_range / n) * upper_entropy;
            double cie = lower_cie + upper_cie;
            if (cie < best_cie) {
                best_cie = cie;
                best_lower_entropy = lower_entropy;
                best_upper_entropy = upper_entropy;
                best_freq_lower = freq_lower;
                best_freq_upper = freq_upper;
                best_index = i;
            }
        }
    }
    return std::make_tuple(best_index, best_cie, best_lower_entropy, best_upper_entropy, best_freq_lower, best_freq_upper);
}

bool MDLP::mdlpc_criterion(const std::tuple<size_t, double, double, double, std::unordered_map<size_t, size_t>, std::unordered_map<size_t, size_t>> &bests,
                     std::vector<size_t>::iterator start,
                     std::vector<size_t>::iterator end) {
    double n = (double) std::distance(start, end);
    size_t best_index = std::get<0>(bests);
    double cie = std::get<1>(bests);
    double lower_entropy = std::get<2>(bests);
    double upper_entropy = std::get<3>(bests);
    std::unordered_map<size_t, size_t> freq_lower = std::get<4>(bests);
    std::unordered_map<size_t, size_t> freq_upper = std::get<5>(bests);
    std::unordered_map<size_t, size_t> all_freq = count_frequencies(start, end);
    size_t k = all_freq.size(); size_t k1 = freq_lower.size(); size_t k2 = freq_upper.size();
    double y_entropy = entropy(start, end, all_freq);
    double gain = y_entropy - cie;
    double delta = log(pow(3, k) - 2) - (k * y_entropy - k1 * lower_entropy - k2 * upper_entropy);
    return gain > log(n - 1) / n + delta / n;
}

std::forward_list<size_t> MDLP::part(std::vector<double> &x, std::vector<size_t> &y, size_t start, size_t end) {
    if (end - start > 2) {
        auto cut_tuple = find_best_split(x, y, start, end);
        if (mdlpc_criterion(cut_tuple, y.begin() + start, y.begin() + end)) {
            auto cut_point = std::get<0>(cut_tuple);
            auto l1 = part(x, y, start, cut_point + 1);
            auto l2 = part(x, y, cut_point + 1, end);
            l1.splice_after(l1.before_begin(), l2);
            l1.push_front(cut_point);
            return l1;
        } else {
            std::forward_list<size_t>l;
            return l;
        }
    } else {
        std::forward_list<size_t>l;
        return l;
    }
}

std::vector<double> MDLP::mdlp_implementation(std::vector<double> x, std::vector<size_t> y) {
    auto indices = Utils::order(x);
    auto ox = Utils::apply_index(x, indices);
    auto oy = Utils::apply_index(y, indices);
    size_t start = 0;
    size_t end = oy.size();
    auto cut_points = part(ox, oy, start, end);
    std::vector<double> result;
    for (auto cut_point : cut_points) {
        result.push_back((ox[cut_point] + ox[cut_point + 1]) / 2);
    }
    std::sort(result.begin(), result.end());
    return result;
}

//// [[Rcpp::export]]
//Rcpp::List mdlpCpp(Rcpp::DataFrame x, Rcpp::IntegerVector y) {
//    auto y_col = Rcpp::as<std::vector<size_t>>(y);
//    Rcpp::List result = Rcpp::no_init(x.size());
////    std::vector<std::future<std::vector<double>>> fut(x.size());
//    for (size_t i = 0; i < x.size(); i++) {
//        auto x_col = Rcpp::as<std::vector<double>>(x[i]);
////        fut[i] = std::async(mdlp_implementation, x_col, y_col);
//        std::vector<double> cut_points = mdlp_implementation(x_col, y_col);
//        Rcpp::NumericVector vec = Rcpp::wrap(cut_points);
//        result[i] = vec;
//    }
////    for (size_t i = 0; i < x.size(); i++) {
////        result[i] = Rcpp::wrap(fut[i].get());
////    }
//    return result;
//}
