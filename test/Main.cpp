#include "Evolution/StateFlow.hpp"
#include "Evolution/TBBFlow.hpp"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/flow_graph.h"
#include <iostream>

using namespace Evolution;

void AAA() {
  auto sf = StateFlow{};
  auto s0 = sf.GetOrAddInitialState(0);
  auto s1 = sf.GetOrAddInitialState(1);
  auto s2 = sf.AddCrossover(s0, s1);
  sf.SetEvaluate(s0);
  sf.SetEvaluate(s2);

  auto Evaluate = [](int x) { return x * 1.0; };
  auto Mutate = [](int x) { return x + 1; };
  auto Crossover = [](int x, int y) { return x + y; };

  auto flow = TBBFlow(std::move(sf), std::move(Evaluate), std::move(Mutate),
                      std::move(Crossover), []() { return 1; });
  flow.AllowSwapArgumentsInCrossover();
  flow.Run();
  flow.Run();
  flow.Run();
  flow.Run();
  auto const &p = flow.GetPopulation();
  int x = 0;
  x++;
}

int main() {
  AAA();
  std::cout << "aaa" << std::endl;
  return 0;
}
// PATH=C:\Program Files\Intel TBB\tbb\bin\intel64\vc14