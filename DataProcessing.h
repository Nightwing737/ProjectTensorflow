#include "All.h"

class Data{
    std::array<std::vector<float>, 4> TrainTestSplit(Tensor train, Tensor test, float testSplit);
};