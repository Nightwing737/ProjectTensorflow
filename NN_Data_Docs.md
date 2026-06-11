# Data Class

### _Standard Data Processing_

### Data::TrainTestSplit

```cpp
std::array<std::vector<float>,4> Data::TrainTestSplit(Tensor x, Tensor y, float testSplit)
```
Splits input feature tensor x and label tensor y into training and testing sets based on a given testSplit ratio.

Behavior
Extracts raw data from x and y into vectors.
Independently shuffles both feature and label vectors using a random seed.
Computes training size as (1 - testSplit) of the total dataset.
Splits both vectors into:
Training subset
Testing subset

Output
Returns an array of four vectors: xTrain, xTest, yTrain, yTest

---

# Neural Net and Layer Class

### _Neural Net_

### NeuralNet::NeuralNet

```cpp
NeuralNet::NeuralNet(const std::vector<int>& topology)
```
Initializes a feedforward neural network using a layer topology definition.

Accepts a vector topology, where each element defines the number of neurons in a layer.
Iterates through consecutive pairs of layers:
topology[i] → input size
topology[i+1] → output size
For each pair, constructs a layer and appends it to the network’s layers container.

Example:
```cpp
NeuralNet N({3,5,2});     
// Topology: 3 -> 5 -> 2
```

### NeuralNet::forward

```cpp
Tensor NeuralNet::forward(const Tensor& input)
```

Internal function only.
Returns Tensor after a single Forward Pass.

Takes an input Tensor.
Initializes activation A with the input.
Sequentially propagates A through each layer in layers.
Each layer transforms the tensor via its own forward function.
Linear Transform: Z = XW+B
Activation: RELU

### NeuralNet::dMSE

```cpp
Tensor NeuralNet::dMSE(const Tensor& pred, const Tensor& target)
```

Calculates the gradient of MSE loss with respect to predictions

Operation: 2(predicted[i]-actual[i])/N

N = No of elements
Throws error if shapes differ
Outputs Tensor of same shape as input with loss gradients

### NeuralNet::predict
```cpp
Tensor NeuralNet::predict(const Tensor& input)
```
Runs inference for forward pass
Sequentially goes through all layers of NeuralNet 
Outputs Tensor with final predictions

### NeuralNet::Train
```cpp
void NeuralNet::Train(const std::vector<Tensor> &xTrain, const std::vector<Tensor> &yTrain, int epochs, float learnRate)
```
Trains NeuralNet using simple Schoastic Gradient Descent
Throws error if datasize mismatch


### _Layer_

### Layer::Layer

```cpp
Layer::Layer(int inputSize, int outputSize)
```
Internal function only.
Initialises fully connected layer for NN.
Loads Random Weights and (Initialised) Zero Biases. 


### Layer::forward
```cpp
Layer::forward(const Tensor& input)
```
Internal function only.
Forward Propagation for one layer.
inputCache: Input for backprop 
outputCache: Pre-Activation Input

Linear Transform: Z = XW+B
Activation: RELU

### Layer::back
```cpp
Tensor Layer::back(const Tensor& gradOutput, float learnRate)
```
Internal function only.
Performs backpropagation through layers and updates parameters
gradOutput: Gradient from next layer
learnRate: Learning Rate (n)

dZ = dA * dReLU(Z)

dW = transpose(X) * dZ

dB = Rowsum(dZ)

dX = dZ * transpose(W)
W <- W - ndW, B <- B - ndB

---