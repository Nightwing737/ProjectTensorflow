

# Tensor Class
### _Tensor Basics Functions_
### Tensor::Tensor
```cpp
Tensor::Tensor(std::vector<float> data, std::vector<int> shape)
```

Takes a float type vector as data along with shape parameter to define and return multidimensional array (Tensor).
Also sets internal variable "ismat" to true if input array is a matrix. Throws error if vector cannot be represented in given shape.

Usage: 
```cpp
Tensor x({1,2,3,4},{2,2});  
```


### Tensor::element
```cpp
const float element(std::vector<int> index)
```

Accepts int type vector and returns element located at given index. Effectively functions as the [ ] operator for arrays. Uses strides to convert from index array to index for single dimensional vector.

Usage: 
```cpp
Tensor x({1,2,3,4},{2,2});     // {{1,2},{3,4}};
float y = x.element({1,0})      // 3.0
```

### Tensor::getShape
```cpp
std::vector<int> getShape()
```

Returns shape of Tensor object.

Usage:
```cpp
Tensor x({1,2,3,4},{2,2});
std::vector<int> y = x.getShape()       // std::vector({2,2})
```

### Tensor::getData
```cpp
std::vector<float> getData()
```

Returns the 1D vector representation of data held in Tensor object.

Usage:
```cpp
Tensor x({1,2,3,4},{2,2});
std::vector<float> y = x.getShape()  // std::vector({1,2,3,4})
```

### Tensor::getSize
```cpp
int Tensor::getSize()
```

Returns the number of elements in 1D vector representation of data held in Tensor object.

Usage:
```cpp
Tensor x({1,2,3,4},{2,2});
int y = x.getSize()     // 4
```

### Tensor::change
```cpp
void Tensor::change(float val, const std::vector<int>& index);
```
Changes value located at index vector to input value. 

Usage:
```cpp
Tensor a({1,2,3,4},{2,2});
a.change(5,{1,1});      // {1,2,3,5} shape = {2,2}
```
___

### _Standard Tensor Operators_
### Operator "+" 
```cpp
Tensor Tensor::operator+(const Tensor& a)
Tensor Tensor::operator+(float a)
```
Returns the sum of two tensors of same dimensions. Throws error if dimensions of input tensors do not match.
If second value is scalar, adds Tensor value to all elements of Tensor.

Usage
```cpp
Tensor a({1,2,3,4},{2,2});
Tensor b({5,6,7,8},{2,2});
Tensor c({9,8,10},{3,1});

auto x = a + b      // {6,8,10,12} shape = {2,2};
auto y = a + c      // Invalid 
auto z = a + 2      // {3,4,5,6} shape = {2,2};
```

### Operator "-"
```cpp
Tensor Tensor::operator-(const Tensor& a)
Tensor Tensor::operator-(float a)
```
Identical to Operator "+" but for subtraction.

### Operator "*"
```cpp
Tensor Tensor::operator*(const Tensor& a)
Tensor Tensor::operator*(float scalar) const
```
Returns the elementwise product of to tensors of same dimensions. Throws error if dimensions of input tensors do not match.
If second value is scalar, multiples all values in tensor with scalar value.

Usage:
```cpp
Tensor a({1,2,3,4},{2,2});
Tensor b({1,2,3,4},{2,2});
auto c = a * b      // {1,4,9,16} shape = {2,2};
auto d = a * 2      // {2,4,6,8} shape = {2,2};
```


### Tensor::transpose
```cpp
Tensor Tensor::transpose(std::vector<int> perm)
```
Swaps axises based on input vector perm. Index axis is swapped with value at index axis for all elements in perm. Need not change all axes.

Usage:
```cpp
Tensor a({/*data*/},{2,3,4});
auto b = a.transpose({/*data*/},{1,0,2});    // Shape: {3,2,4}
auto c = a.transpose({2});      // Shape: {4,2,3}
```

### Tensor::reshape
```cpp
void Tensor::reshape(const std::vector<int>& newShape)
```
Changes Shape. Throws error if shape is impossible to assign.

Usage:
```cpp
Tensor a({1,2,3,4},{2,2});
a.reshape({4,1});   // Shape: {4,1}
```

### Tensor::Rowsum
```cpp
Tensor Tensor::Rowsum() const
```
Returns Sum of Rows of Tensor as a Tensor.

Usage:
```cpp
Tensor a({1,2,3,4,5,6,7,8,9,10,11,12}, {3,4});
auto b = a.Rowsum();
```

---
### _Matrix specific functions_
### Tensor::isMat
```cpp
int Tensor::isMat() const
```
Returns 1 if Tensor is Matrix (shape is 2 dimensional). Else 0.

### Tensor::matmul
```cpp
Tensor Tensor::matmul(const Tensor& a)
```
Performs Matrix Multiplication on two given Matrices. Throws error if dimensions incompatible to do Matrix Multiplication.

Usage:
```cpp
Tensor x({1,2,3,4},{2,2});
Tensor y({5,6,7,8},{2,2});
auto z = x.matmul(y);    // {19,22,43,50} shape = {2,2};
```

### Tensor::det
```cpp
float Tensor::det() const
```
Finds the determinant of a given Matrix using LU Decomposition. Throws error if given Matrix is not a square Matrix. May need to round result. 

Usage:
```cpp
Tensor x({1,2,3,4},{2,2});
float d = x.det();      // -1.99999988
```
---

### _Random Vector Initializations_

### Tensor::RandomTensor
```cpp
static Tensor Tensor::RandomTensor(std::vector<int> shape, int start, int end, int floatFlag)
```
Returns a Tensor with random values and specified shape. Random values range from start to end. If floatFlag = truem float random numbers allowed. Otherwise only integer random numbers.

Usage:
```cpp
auto x = Tensor::RandomTensor({4,5,6},0,10);
```

### Tensor::ZeroTensor
```cpp
static Tensor Tensor::ZeroTensor(std::vector<int> shape)
```
Same as RandomTensor but returns Tensor with only Zeroes. Only shape
parameter needed.

---

### _Miscellaneous_

Added Custom Tensor operations specifically to set up/make Neural Net computation easier.

### Tensor::Relu
```cpp
Tensor Tensor::Relu() const
```
RELU Activtion function

### Tensor:dRelu
```cpp
Tensor Tensor::dRelu() const
```
Gradient of RELU Activation function

---