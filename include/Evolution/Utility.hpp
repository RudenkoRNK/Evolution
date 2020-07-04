#pragma once
#define NOMINMAX
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace Evolution {

template <class T, class Indexer, class IndexFunction>
inline void Permute(std::vector<T> &v, std::vector<Indexer> &perm,
                    IndexFunction &&Index) {
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
  assert(&v != &perm);
#endif // !NDEBUG

  auto control = std::vector<size_t>(v.size());
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

} // namespace Evolution
