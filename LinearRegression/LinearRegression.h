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

class LinReg{
    private:
    Tensor x = Tensor::ZeroTensor({1,1});
    Tensor y = Tensor::ZeroTensor({1,1});
    int m;
    int b;

    LinReg(const Tensor& input);
    float MSE(const Tensor& pred, const Tensor& target);
    float mGrad(const Tensor& pred, const Tensor& target);
    float bGrad(const Tensor& pred, const Tensor& target);
    Tensor predict();
    void Train(const Tensor& target, float lr);
};