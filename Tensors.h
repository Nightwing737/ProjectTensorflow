#pragma once
#include "All.h"

class Tensor{
    private:
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> strides;
    int ismat;

    Tensor(std::vector<float> data, std::vector<int> shape);
    void computeStrides();
    int flatIndex(const std::vector<int>& index);

    public:
    int isMat() const;
    static Tensor constant(std::vector<float> data, std::vector<int> shape);
    const float element(const std::vector<int>& index);
    void change(float val, const std::vector<int>& index);
    Tensor operator+(const Tensor& a);
    Tensor operator+(float a);
    Tensor operator-(const Tensor& a);
    Tensor operator-(float a);
    Tensor mul(float scalar);
    Tensor matmul(const Tensor& a);
    float det() const;
    Tensor transpose(std::vector<int> perm);
    void print();
    std::vector<int> getShape();
    std::vector<float> getData();
    int getSize();
    void modify(std::vector<float> newData);
    static std::vector<float> RandomVec(int start, int end, int n, int floatflag);
    static Tensor RandomTensor(std::vector<int> shape, int start, int end, int f);
};