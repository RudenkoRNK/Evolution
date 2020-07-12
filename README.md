# Evolution
A header-only library implementing genetic algorithm.

This library allows flexible set up of GA, suggesting simple interface for generating sequence of mutates and crossovers.

The parallelism is achieved with Intel TBB library.

## How to build tests
cmake -DBOOST_ROOT=\<path to boost\> -DTBB_DIR=\<path to Intel TBB cmake directory\> -DEvolution_BUILD_TEST=ON \<path to sources\>

For Windows you may need to add path to Intel TBB libraries:

PATH+="C:/Program Files/Intel TBB/tbb/bin/intel64/vc14"
