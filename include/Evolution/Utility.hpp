#pragma once
#define NOMINMAX
#include "Evolution/TypeTraits.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <vector>

namespace Evolution {

inline std::vector<size_t> GetIndices(size_t size) {
  auto indices = std::vector<size_t>(size);
  std::iota(indices.begin(), indices.end(), size_t{0});
  return indices;
}

template <class T>
inline void Permute(std::vector<T> &v, std::vector<size_t> &perm) {
  Permute(v, perm, std::identity{});
}

template <class T, class Indexer, class IndexFunction>
inline void Permute(std::vector<T> &v, std::vector<Indexer> &perm,
                    IndexFunction &Index) {
#ifndef NDEBUG
  assert(std::unique(perm.begin(), perm.end(),
                     [&](Indexer const &lhs, Indexer const &rhs) {
                       return Index(lhs) == Index(rhs);
                     }) == perm.end());
  assert(*std::min_element(perm.begin(), perm.end(),
                           [&](Indexer const &lhs, Indexer const &rhs) {
                             return lhs < rhs;
                           }) == 0);
  assert(*std::max_element(perm.begin(), perm.end(),
                           [&](Indexer const &lhs, Indexer const &rhs) {
                             return lhs < rhs;
                           }) == perm.size() - 1);
  if constexpr (std::is_same_v<T, Indexer>) {
    assert(&v != &perm);
  }
#endif // !NDEBUG

  auto &&control = std::vector<size_t>(v.size());
  std::iota(control.begin(), control.end(), size_t{0});
  for (auto i = size_t{0}, e = v.size(); i < e; ++i) {
    while (Index(perm.at(i)) != i) {
      std::swap(control.at(i), control.at(Index(perm.at(i))));
      std::swap(perm.at(i), perm.at(Index(perm.at(i))));
    }
  }
  for (auto i = size_t{0}, e = v.size(); i < e; ++i) {
    while (control.at(i) != i) {
      std::swap(v.at(i), v.at(control.at(i)));
      std::swap(perm.at(i), perm.at(control.at(i)));
      std::swap(control.at(i), control.at(control.at(i)));
    }
  }
}

template <class FG, class... Args>
inline std::chrono::nanoseconds Benchmark(FG &&Func, Args &&... args) {
  auto start = std::chrono::steady_clock::now();
  Func(std::forward<Args>(args)...);
  auto end = std::chrono::steady_clock::now();
  return end - start;
}

template <class Number> inline double Mean(std::vector<Number> const &v) {
  auto sum = std::accumulate(v.begin(), v.end(), 0.0);
  return sum / v.size();
}

template <class Number> inline double Variance(std::vector<Number> const &v) {
  auto mean = Mean(v);
  auto sum = std::accumulate(v.begin(), v.end(), 0.0, [mean](auto x, auto y) {
    return x + (y - mean) * (y - mean);
  });
  return sum / v.size();
}

// Root mean square
template <class Number> inline double RMS(std::vector<Number> const &v) {
  auto sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
  return sqrt(sum / v.size());
}

struct LinearFit {
  // y = a + b * x
  double a;
  double b;
  double sigma_a;
  double sigma_b;
};

template <class NumberX, class NumberY>
inline LinearFit LeastSquares(std::vector<NumberX> const &x,
                              std::vector<NumberY> const &y) {
  assert(x.size() == y.size());
  assert(y.size() >= 2);
  auto n = y.size();
  auto xAvg = Mean(x);
  auto yAvg = Mean(y);
  auto x2Avg = pow(RMS(x), 2);
  auto y2Avg = pow(RMS(y), 2);
  auto xyAvg = std::inner_product(x.begin(), x.end(), y.begin(), 0.0) / n;
  // y = a + b * x
  auto b = (xyAvg - xAvg * yAvg) / (x2Avg - xAvg * xAvg);
  auto a = yAvg - b * xAvg;
  auto sigma_b =
      sqrt((y2Avg - yAvg * yAvg) / (x2Avg - xAvg * xAvg) - b * b) / sqrt(n);
  auto sigma_a = sigma_b * sqrt(x2Avg - xAvg * xAvg);
  return {.a = a, .b = b, .sigma_a = sigma_a, .sigma_b = sigma_b};
}

template <class Number>
inline LinearFit LeastSquares(std::vector<Number> const &y) {
  auto x = std::vector<double>(y.size());
  std::iota(x.begin(), x.end(), 0.0);
  return LeastSquares(x, y);
}

} // namespace Evolution
