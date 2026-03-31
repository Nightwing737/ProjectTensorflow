#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <chrono> //* Testing purposes
#include <thread> //* Testing purposes

class Tensor{
    private:
    std::vector<int> shape;
    std::vector<float> data;
    int ismat;
    
    Tensor(std::vector<float> data, std::vector<int> shape) : data(std::move(data)), shape(std::move(shape)) {
        ismat = (this->shape.size() == 2);
    };
    
    public:

    int isMat() const{
        return ismat;
    }

    static Tensor constant(std::vector<float> data, std::vector<int> shape){
        int expected = 1;
        for (auto d : shape){
            if (d <= 0){
                throw std::invalid_argument("Shape must be positive");
            }
            expected *= d;
        }
        if (data.size() != expected){
            throw std::invalid_argument("Dimensions do not match");
        }
        return Tensor{std::move(data),std::move(shape)};
    }

    const float element(std::vector<int> index){
        if (index.size() != shape.size()){
            throw std::invalid_argument("Dimensions do not match");
        }
        int n = shape.size();
        
        int flat_index = 0;
        std::vector<int> strides;
        int stride = 1;

        for (int x : shape){
            stride *= x;
        }

        for (int i = 0; i < shape.size(); i++){
            if (index[i] < 0 || index[i] >= shape[i]){
                throw std::invalid_argument("Index out of bounds");
            }
            stride = stride/shape[i];
            flat_index += index[i] * stride;
        }
        return data[flat_index];
    }

    Tensor operator+(const Tensor& a){
        
        if (shape != a.shape){
            throw std::invalid_argument("Dimensions do not match");
        }

        std::vector<float> result(data.size());

        for (int i = 0; i < data.size(); i++){
            result[i] = data[i] + a.data[i];
        }
        return Tensor(result, shape);
    }

    Tensor operator-(const Tensor& a){
        if (shape != a.shape){
            throw std::invalid_argument("Dimensions do not match");
        }

        std::vector<float> result(data.size());

        for (int i = 0; i < data.size(); i++){
            result[i] = data[i] - a.data[i];
        }
        return Tensor(result, shape);
    }

    Tensor mul(float scalar){

        std::vector<float> result(data.size());
        for (int i = 0; i < data.size(); i++){
            result[i] = data[i] * scalar;
        }
        return Tensor(result,shape);
    }

    Tensor matmul(const Tensor& a){
        if (ismat && a.ismat){
            if (shape[1] == a.shape[0]){
                int m = shape[0];
                int n = shape[1];
                int p = a.shape[1];
                std::vector<float> result(m*p,0);

                for (int i = 0; i < m; i++){
                    for (int j = 0; j < p; j++){
                        for (int k = 0; k < n; k++){
                            float A = data[i*n+k];
                            float B = a.data[k*p+j];

                            result[i*p+j] += A*B;
                        }
                    }
                }
                return Tensor(result, {m,p});
            }
            throw std::invalid_argument("Matrix dimensions incompatible");
        }
        throw std::invalid_argument("Cannot multiply non matrices");
    }

    float det() const{
        if (ismat){
            if (shape[0] == shape[1]){
                int n = shape[0];

                std::vector<float> copy = data;

                float det = 1;
                int sign = 1;

                for (int i = 0; i < n; i++){
                    int pivot = i;
                    for(int j = i+1; j < n; j++){
                        if (std::abs(copy[j*n+i])>std::abs(copy[pivot*n+i])){
                            pivot = j;
                        }
                    }
                    if (std::abs(copy[pivot*n+i]) < 1e-6){
                        return 0;
                    }
                    if (pivot != i){
                        for (int j = 0; j < n; j++){
                            std::swap(copy[i*n+j],copy[pivot*n+j]);
                        }
                        sign *= -1;
                    }

                    float pivotVal = copy[i*n+i];

                    for (int j = i+1; j < n; j++){
                        float factor = copy[j*n+i]/pivotVal;

                        for (int k = i; k < n; k++){
                            copy[j*n+k] -= factor *copy[i*n+k];
                        }
                    }
                    det *= pivotVal; 
                }
                return det*sign;
            }
            throw std::invalid_argument("Cannot find determinant of non square matrix");
        }
        throw std::invalid_argument("Cannot find determinant of non matrices");
    }

    void print(){
        //TODO: Implement
    }
};


int main(){
    // Testing purposes
}