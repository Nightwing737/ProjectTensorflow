#include "Tensors.h"
/*
Constructor: Takes a float type vector as data along with shape parameter to define and return multidimensional array (Tensor).
Also sets internal variable "ismat" to true if input array is a matrix. Throws error if vector cannot be represented in given shape.
*/
Tensor::Tensor(std::vector<float> data, std::vector<int> shape) : data(std::move(data)), shape(std::move(shape)) {
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

    ismat = (this->shape.size() == 2);
    computeStrides();
};

/*
Returns 1 if Tensor is Matrix (shape is 2 dimensional). Else 0.
*/
int Tensor::isMat() const{
    return ismat;
}

void Tensor::computeStrides(){
    strides.resize(shape.size());

    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--){
        strides[i] = stride;
        stride *= shape[i];
    } 
    
}

int Tensor::flatIndex(const std::vector<int>& index){
    int flat_index = 0;

    for (int i = 0; i < shape.size(); i++){
        if (index[i] < 0 || index[i] >= shape[i]){
            throw std::invalid_argument("Index out of bounds");
        }
        flat_index += index[i] * strides[i];
    }
    return flat_index;
}

/*
Accepts int type vector and returns element located at given index. 
Effectively functions as the [ ] operator for arrays. Uses strides to convert from index array to index for single dimensional vector.
*/
const float Tensor::element(const std::vector<int>& index){
    if (index.size() != shape.size()){
        throw std::invalid_argument("Dimensions do not match");
    }
    
    int flat_index = flatIndex(index);
    return data[flat_index];
}

/*
Changes value located at index vector to input value. 
*/
void Tensor::change(float val, const std::vector<int>& index){
    if (index.size() != shape.size()){
        throw std::invalid_argument("Dimensions do not match");
    }

    int flat_index = flatIndex(index);
    data[flat_index] = val;
}

/*
Returns the sum of two tensors of same dimensions. Throws error if dimensions of input tensors do not match.
*/
Tensor Tensor::operator+(const Tensor& a) const{
    
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
Adds constant value to all elements of Tensor.
*/
Tensor Tensor::operator+(float a) const{
    Tensor result = *this;
    
    for (int i = 0; i < result.data.size(); i++){
        result.data[i] += a;
    }
    return *this;
}

/*
Returns the difference of two tensors of same dimensions. Throws error if dimensions of input tensors do not match.
*/
Tensor Tensor::operator-(const Tensor& a) const{
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
Subtracts constant value from all elements of Tensor.
*/
Tensor Tensor::operator-(float a) const{
    Tensor result = *this;

    for (int i = 0; i < result.data.size(); i++){
        result.data[i] -= a;
    }
    return *this;
}



/*
Multiplies all values in tensor with input scalar value.
*/
Tensor Tensor::operator*(float scalar) const{

    std::vector<float> result(data.size());
    for (int i = 0; i < data.size(); i++){
        result[i] = data[i] * scalar;
    }
    return Tensor(result,shape);
}

Tensor Tensor::operator*(const Tensor& a) const{

    std::vector<float> result(data.size());
    for (int i = 0; i < data.size(); i++){
        result[i] = data[i]*a.data[i];
    }
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

Tensor Tensor::transpose(std::vector<int> perm){
    if (perm.size() > shape.size()){
        throw std::invalid_argument("Invalid permutation size");
    }

    std::vector<int> used(shape.size(), 0);

    for (int p: perm){
        if (p < 0 || p >= shape.size() || used[p]){
            throw std::invalid_argument("Invalid or duplicate axis");
        }
        used[p] = 1;
    }

    std::vector<int> fullPerm = perm;
    for (int i = 0; i < shape.size(); i++) {
        if (!used[i]) {
            fullPerm.push_back(i);
        }
    }

    std::vector<int> newShape(shape.size());

    for (int i = 0; i < shape.size(); i++){
        newShape[i] = shape[fullPerm[i]];
    }

    Tensor result(std::vector<float>(data.size(),0),newShape);
    result.computeStrides();

    const std::vector<int>& oldStrides = strides;
    const std::vector<int>& newStrides = result.strides;

    for (int i = 0; i < data.size(); i++){
        int t = i;
        int oldIndex = 0;

        for (int i = 0; i < shape.size(); i++){
            int coord = t/newStrides[i];
            t %= newStrides[i];

            oldIndex += coord * oldStrides[fullPerm[i]];
        }

        result.data[i] = data[oldIndex];
    }
    return result;

}

void Tensor::print(){
    //TODO: Implement

    
}

/*
Returns shape of Tensor object.
*/
const std::vector<int>& Tensor::getShape() const{
    return shape;
}

/*
Returns the 1D vector representation of data held in Tensor object.
*/
const std::vector<float>& Tensor::getData() const{
    return data;
}

/*
Returns the number of elements in 1D vector representation of data 
held in Tensor object.
*/
int Tensor::getSize() const{
    return data.size();
}

/*
Chnage Shape
*/
void Tensor::reshape(const std::vector<int>& newShape){
    int newSize = 1;
    for (int x : newShape){
        if (x <= 0){
            throw std::invalid_argument("Invalid shape dim");
        }
        newSize *= x;
    }

    if (newSize != data.size()){
        throw std::invalid_argument("Reshape must contain same no of elements");
    }
    shape = newShape;
    computeStrides();
}

/*
Returns Sum of Rows of Tensor
*/
Tensor Tensor::Rowsum() const{
    if (shape.size() != 2) {
        throw std::invalid_argument("sum_rows requires 2D tensor");
    }

    int rows = shape[0];
    int cols = shape[1];

    std::vector<float> result(cols, 0.0f);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j] += data[i * cols + j];
        }
    }

    return Tensor(std::move(result), {1, cols});
}

/*
Returns a Tensor with random values and specified shape. Random values range from start to end. 
If floatFlag = truem float random numbers allowed. Otherwise only integer random numbers.
*/
Tensor Tensor::RandomTensor(std::vector<int> shape, float start, float end){
    int n = 1;
    for (int x : shape){
        n*=x;
    }
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(start, end);
    
    std::vector<float> v(n);
    for (auto &x : v) x = dist(gen);

    return Tensor(std::move(v), std::move(shape));
}

/*
Returns a Tensor of just zeroes of given shape
*/
Tensor Tensor::ZeroTensor(std::vector<int> shape){
    int n = 1;
    for (int x : shape){
        n*=x;
    }
    std::vector<float> v(n, 0);
    return Tensor(v, shape);
}   

Tensor Tensor::Relu() const{
    std::vector<float> result(data.size());

    for (int i = 0; i < data.size(); i++){
        if (data[i] > 0){
            result[i] = data[i];
        }
        else{
            result[i] = 0;
        }
    }

    return Tensor(std::move(result),shape);
}

Tensor Tensor::dRelu() const{
    std::vector<float> result(data.size());

    for (int i = 0; i < data.size(); i++){
        if (data[i] > 0){
            result[i] = 1.0;
        }
        else{
            result[i] = 0;
        }
    }
    return Tensor(std::move(result),shape);
}


int main(){
    // Testing purposes
    Tensor t = Tensor::RandomTensor({4,5,6},0,10);
    t.element({1,3,2});
    Tensor t2 = Tensor::ZeroTensor({3,2});
}