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

class Tensor{
    private:
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> strides;
    int ismat;

    void computeStrides();
    int flatIndex(const std::vector<int>& index);
    
    public:
    Tensor(std::vector<float> data, std::vector<int> shape);
    int isMat() const;
    const float element(const std::vector<int>& index);
    void change(float val, const std::vector<int>& index);
    Tensor operator+(const Tensor& a);
    Tensor operator+(float a);
    Tensor operator-(const Tensor& a);
    Tensor operator-(float a);
    Tensor operator*(float scalar);
    Tensor matmul(const Tensor& a);
    float det() const;
    Tensor transpose(std::vector<int> perm);
    void print();
    const std::vector<int>& getShape() const;
    const std::vector<float>& getData() const;
    int getSize() const;
    void Tensor::reshape(const std::vector<int>& newShape);
    static Tensor RandomTensor(std::vector<int> shape, float start, float end);
    static Tensor ZeroTensor(std::vector<int> shape);
    Tensor Rowsum() const;
    Tensor Relu() const;
    Tensor dRelu() const;
};