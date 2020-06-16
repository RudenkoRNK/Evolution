# Evolution
A header-only library implementing genetic algorithm.

This library allows flexible set up of GA, suggesting simple interface for generating sequence of mutates and crossovers.

The parallelism is achieved with Intel TBB library.

## How to build tests
cmake -DBOOST_ROOT=\<path to boost\> -DTBB_DIR=\<path to Intel TBB cmake directory\> \<path to sources\>