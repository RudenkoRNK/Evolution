#pragma once
#include "StateFlow.hpp"
#include "TBBFlow.hpp"

namespace Evolution {

template <class EvaluateFG, class MutateFG, class CrossoverFG>
class Evolution final {
private:
  size_t nElites;
  size_t nMutate;
  size_t nCrossover;

public:
};

} // namespace Evolution

namespace Evolution {} // namespace Evolution
