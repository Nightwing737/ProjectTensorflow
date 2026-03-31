

# Tensor Class
### _Tensor Basics Functions_
### Tensor::constant
```cpp
static Tensor constant(std::vector<float> data, std::vector<int> shape)
```

Takes a float type vector as data along with shape parameter to define and return multidimensional array (Tensor).
Also sets internal variable "ismat" to true if input array is a matrix. Throws error if vector cannot be represented in given shape.

Usage: 
```cpp
auto x = Tensor::constant({1,2,3,4},{2,2});  
```


### Tensor::element
```cpp
const float element(std::vector<int> index)
```

Accepts int type vector and returns element located at given index. Effectively functions as the [ ] operator for arrays. Uses strides to convert from index array to index for single dimensional vector.

Usage: 
```cpp
auto x = Tensor::constant({1,2,3,4},{2,2});     // {{1,2},{3,4}};
float y = x.element({1,0})      // 3.0
```
___

### _Standard Tensor Operators_
### Operator "+" 
```cpp
Tensor operator+(const Tensor& a)
```
Returns the sum of two tensors of same dimensions. Throws error if dimensions of input tensors do not match.

Usage
```cpp
auto a = Tensor::constant({1,2,3,4},{2,2});
auto b = Tensor::constant({5,6,7,8},{2,2});
auto c = Tensor::constant({9,8,10},{3,1});

auto x = a + b      // {6,8,10,12} shape = {2,2};
auto y = a + c      // Invalid 
```

### Operator "-"
```cpp
Tensor operator-(const Tensor& a)
```
Identical to Operator "+" but for subtraction.

### Tensor::mul
```cpp
Tensor mul(float scalar)
```
Multiplies all values in tensor with input scalar value.

Usage:
```cpp
auto a = Tensor::constant({1,2,3,4},{2,2});
auto b = a.mul(5)       // {5,10,15,20};
```

---
### _Matrix specific functions_
### Tensor::isMat
```cpp
int isMat() const
```
Returns 1 if Tensor is Matrix (shape is 2 dimensional). Else 0.

### Tensor::mul
```cpp
Tensor matmul(const Tensor& a)
```
Performs Matrix Multiplication on two given Matrices. Throws error if dimensions incompatible to do Matrix Multiplication.

Usage:
```cpp
auto x = Tensor::constant({1,2,3,4},{2,2});
auto y = Tensor::constant({5,6,7,8},{2,2});
auto z = x.matmul(y);    // {19,22,43,50} shape = {2,2};
```

### Tensor::det
```cpp
float det() const
```
Finds the determinant of a given Matrix using LU Decomposition. Throws error if given Matrix is not a square Matrix. May need to round result. 

Usage:
```cpp
auto x = Tensor::constant({1,2,3,4},{2,2});
float d = x.det();      // -1.99999988
```
---