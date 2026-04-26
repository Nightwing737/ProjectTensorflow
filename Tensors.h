#include "All.h"

class Tensor{
    private:
    std::vector<int> shape;
    std::vector<float> data;
    int ismat;

    Tensor(std::vector<float> data, std::vector<int> shape) : data(std::move(data)), shape(std::move(shape)) {};

    public:
    int isMat() const;
    static Tensor constant(std::vector<float> data, std::vector<int> shape);
    const float element(std::vector<int> index);
    Tensor operator+(const Tensor& a);
    Tensor operator-(const Tensor& a);
    Tensor mul(float scalar);
    Tensor matmul(const Tensor& a);
    float det() const;
    void print();
    std::vector<int> getShape();
    std::vector<float> getData();
    int getSize();
    void modify(std::vector<float> newData);
    std::vector<float> RandomVec(int start, int end, int n, int floatflag);
    Tensor RandomTensor(std::vector<int> shape, int start, int end, int f);
};