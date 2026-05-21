#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <functional>
#include <array>
#include <chrono> //* Testing purposes
#include <thread> //* Testing purposes
#include "Tensors/Tensors.h"

class Data{
    public:
    std::array<std::vector<float>, 4> TrainTestSplit(Tensor train, Tensor test, float testSplit);
};