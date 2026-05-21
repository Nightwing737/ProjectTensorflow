#include "LinearRegression.h"

LinReg::LinReg(const Tensor& input){
    x = input;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1,100);

    m = distr(gen);
    b = distr(gen);
}

float LinReg::MSE(const Tensor& pred, const Tensor& target){

    Tensor T = pred-target;
    T = T*T;

    float J = 0;
    for (float x : T.getData()){
        J += x;
    }
    J = J/T.getSize();
    return J;
}

float LinReg::mGrad(const Tensor& pred, const Tensor& target){

    Tensor T = (pred - target) * x;
    
    float Grad = 0;
    for (float x : T.getData()){
        Grad += x;
    }
    Grad = ((2)/T.getSize()) * Grad;
    return Grad;
}

float LinReg::bGrad(const Tensor& pred, const Tensor& target){
    
    Tensor T = pred - target;

    float Grad;
    for (float x : T.getData()){
        Grad += x;
    }
    Grad = ((2)/T.getSize()) * Grad;
    return Grad;
}

Tensor LinReg::predict(){
    return (x * m) + b;
}

void LinReg::Train(const Tensor& target, float lr){

    Tensor pred = predict();

    float dm = mGrad(pred, target);
    float db = bGrad(pred, target);

    m -= lr * dm;
    b -= lr * db;
}