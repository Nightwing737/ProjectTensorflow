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

class Layer{
    private:
    Tensor Weights = Tensor::ZeroTensor({1,1});
    Tensor Bias = Tensor::ZeroTensor({1,1});

    Tensor inputCache = Tensor::ZeroTensor({1,1});
    Tensor outputCache = Tensor::ZeroTensor({1,1});

    Tensor dWeights = Tensor::ZeroTensor({1,1});
    Tensor dBias = Tensor::ZeroTensor({1,1});
    
    public:
    Layer(int inputSize, int outputSize);

    Tensor forward(const Tensor& input);
    Tensor back(const Tensor& gradOutput, float learnRate); 
};

class NeuralNet{
    private:
    std::vector<Layer> layers;

    public:
    NeuralNet(const std::vector<int> &topology);

    Tensor forward(const Tensor &input);
    Tensor dMSE(const Tensor& pred, const Tensor& target);
    void Train(
        const std::vector<Tensor> &xTrain,
        const std::vector<Tensor> &yTrain,
        int epochs,
        float learnRate
    );

    Tensor predict(const Tensor& input);
};