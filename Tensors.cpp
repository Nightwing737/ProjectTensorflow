#include "Tensors.h"

/*
Constructor
*/
Tensor::Tensor(std::vector<float> data, std::vector<int> shape) : data(std::move(data)), shape(std::move(shape)) {
    ismat = (this->shape.size() == 2);
};

/*
Returns 1 if Tensor is Matrix (shape is 2 dimensional). Else 0.
*/
int Tensor::isMat() const{
    return ismat;
}

/*
Takes a float type vector as data along with shape parameter to define and return multidimensional array (Tensor).
Also sets internal variable "ismat" to true if input array is a matrix. Throws error if vector cannot be represented in given shape.
*/
Tensor Tensor::constant(std::vector<float> data, std::vector<int> shape){
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

/*
Accepts int type vector and returns element located at given index. 
Effectively functions as the [ ] operator for arrays. Uses strides to convert from index array to index for single dimensional vector.
*/
const float Tensor::element(std::vector<int> index){
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
/*
Returns the sum of two tensors of same dimensions. Throws error if dimensions of input tensors do not match.
*/
Tensor Tensor::operator+(const Tensor& a){
    
    if (shape != a.shape){
        throw std::invalid_argument("Dimensions do not match");
    }

    std::vector<float> result(data.size());

    for (int i = 0; i < data.size(); i++){
        result[i] = data[i] + a.data[i];
    }
    return Tensor(result, shape);
}

/*
Returns the difference of two tensors of same dimensions. Throws error if dimensions of input tensors do not match.
*/
Tensor Tensor::operator-(const Tensor& a){
    if (shape != a.shape){
        throw std::invalid_argument("Dimensions do not match");
    }

    std::vector<float> result(data.size());

    for (int i = 0; i < data.size(); i++){
        result[i] = data[i] - a.data[i];
    }
    return Tensor(result, shape);
}

/*
Multiplies all values in tensor with input scalar value.
*/
Tensor Tensor::mul(float scalar){

    std::vector<float> result(data.size());
    for (int i = 0; i < data.size(); i++){
        result[i] = data[i] * scalar;
    }
    return Tensor(result,shape);
}

/*
Performs Matrix Multiplication on two given Matrices. Throws error if dimensions incompatible to do Matrix Multiplication.
*/
Tensor Tensor::matmul(const Tensor& a){
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

/*
Finds the determinant of a given Matrix using LU Decomposition. 
Throws error if given Matrix is not a square Matrix. May need to round result. 
*/
float Tensor::det() const{
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

void Tensor::print(){
    //TODO: Implement
}

/*
Returns shape of Tensor object.
*/
std::vector<int> Tensor::getShape(){
    return shape;
}

/*
Returns the 1D vector representation of data held in Tensor object.
*/
std::vector<float> Tensor::getData(){
    return data;
}

/*
Returns the number of elements in 1D vector representation of data 
held in Tensor object.
*/
int Tensor::getSize(){
    return data.size();
}

/*
Allows for changing of data of a Tensor object to a new vector value. 
Sizes (no of elements) of initial and new data must be same.
*/
void Tensor::modify(std::vector<float> newData){
    if (newData.size() != data.size()){
        throw std::invalid_argument("Shape mismatch");
    }
    data = newData;
}

/*
Generates a random Vector of n elements. Random values range from start to end. 
If floatFlag = true, float random numbers allowed. Otherwise only integer random numbers.
*/
std::vector<float> Tensor::RandomVec(int start, int end, int n, int floatFlag){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<float> v(n);
    if (floatFlag){
        std::uniform_real_distribution<> distrib(start, end);
        std::generate(v.begin(), v.end(), [&](){return distrib(gen);});
    }
    else{
        std::uniform_int_distribution<> distrib(start, end);
        std::generate(v.begin(), v.end(), [&](){return distrib(gen);});
    }
    return v;
}

/*
Returns a Tensor with random values and specified shape. Random values range from start to end. 
If floatFlag = truem float random numbers allowed. Otherwise only integer random numbers.
*/
Tensor Tensor::RandomTensor(std::vector<int> shape, int start, int end, int floatFlag){
    int n = getSize();
    std::vector<float> v = RandomVec(start, end, n, floatFlag);
    return Tensor(v,shape);
}


int main(){
    // Testing purposes
}