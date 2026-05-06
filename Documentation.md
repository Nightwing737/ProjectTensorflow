

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

### Tensor::getShape
```cpp
std::vector<int> getShape()
```

Returns shape of Tensor object.

Usage:
```cpp
auto x = Tensor::constant({1,2,3,4},{2,2});
std::vector<int> y = x.getShape()       // std::vector({2,2})
```

### Tensor::getData
```cpp
std::vector<float> getData()
```

Returns the 1D vector representation of data held in Tensor object.

Usage:
```cpp
auto x = Tensor::constant({1,2,3,4},{2,2});
std::vector<float> y = x.getShape()  // std::vector({1,2,3,4})
```

### Tensor::getSize
```cpp
int Tensor::getSize()
```

Returns the number of elements in 1D vector representation of data held in Tensor object.

Usage:
```cpp
auto x = Tensor::constant({1,2,3,4},{2,2});
int y = x.getSize()     // 4
```

### Tensor::change
```cpp
void Tensor::change(float val, const std::vector<int>& index);
```
Changes value located at index vector to input value. 

Usage:
```cpp
auto a = Tensor::constant({1,2,3,4},{2,2});
a.change(5,{1,1});      // {1,2,3,5} shape = {2,2}
```
___

### _Standard Tensor Operators_
### Operator "+" 
```cpp
Tensor operator+(const Tensor& a)
Tensor operator+(float a)
```
Returns the sum of two tensors of same dimensions. Throws error if dimensions of input tensors do not match.
If second value is scalar, adds constant value to all elements of Tensor.

Usage
```cpp
auto a = Tensor::constant({1,2,3,4},{2,2});
auto b = Tensor::constant({5,6,7,8},{2,2});
auto c = Tensor::constant({9,8,10},{3,1});

auto x = a + b      // {6,8,10,12} shape = {2,2};
auto y = a + c      // Invalid 
auto z = a + 2      // {3,4,5,6} shape = {2,2};
```

### Operator "-"
```cpp
Tensor operator-(const Tensor& a)
Tensor operator-(float a)
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

### Tensor::modify
```cpp
void modify(std::vector<float> newData)
```

Allows for changing of data of a Tensor object to a new vector value. Sizes (no of elements) of initial and new data must be same.

Usage:
```cpp
auto a = Tensor::constant({1,2,3,4},{2,2});
std::vector<float> d = {5,6,7,8};
a.modify(d);    // {5,6,7,8}, Shape = {2,2}
```

### Tensor::transpose
```cpp
Tensor Tensor::transpose(std::vector<int> perm)
```
Swaps axises based on input vector perm. Index axis is swapped with value at index axis for all elements in perm. Need not change all axes.

Usage:
```cpp
auto a = Tensor::constant({/*data*/},{2,3,4});
auto b = a.transpose({/*data*/},{1,0,2});    // Shape: {3,2,4}
auto c = a.transpose({2});      // Shape: {4,2,3}
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

### _Random Vector Initializations_

### Tensor::RandomVec
```cpp
static std::vector<float> RandomVec(int start, int end, int n, int floatFlag)
```

Returns a random Vector of n elements. Random values range from start to end. If floatFlag = true, float random numbers allowed. Otherwise only integer random numbers.

Usage:
```cpp
auto x = Tensor::RandomVec(0,10,6,0);
// x = std::vector({5,3,2,1,10,4}) as float
```

### Tensor::RandomTensor
```cpp
static Tensor RandomTensor(std::vector<int> shape, int start, int end, int floatFlag)
```

Returns a Tensor with random values and specified shape. Random values range from start to end. If floatFlag = truem float random numbers allowed. Otherwise only integer random numbers.


# Data Class

### _Standard Data Processing_

### Data::TrainTestSplit

```cpp
std::array<std::vector<float>,4> Data::TrainTestSplit(Tensor x, Tensor y, float testSplit)
```
W.I.P.
