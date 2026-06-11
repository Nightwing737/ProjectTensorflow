#include "NeuralNet.h"

//* NEURALNET CLASS

/*
Initializes a feedforward neural network using a layer topology definition.

Accepts a vector topology, where each element defines the number of neurons in a layer.
Iterates through consecutive pairs of layers:
topology[i] → input size
topology[i+1] → output size
For each pair, constructs a layer and appends it to the network’s layers container.
*/
NeuralNet::NeuralNet(const std::vector<int>& topology){
    for (int i = 0; i < topology.size()-1; i++){
        int inputSize = topology[i];
        int outputSize = topology[i+1];

        layers.emplace_back(inputSize, outputSize);
    }
}

/*
Returns Tensor after a single Forward Pass

Takes an input Tensor.
Initializes activation A with the input.
Sequentially propagates A through each layer in layers.
Each layer transforms the tensor via its own forward function.

*/
Tensor NeuralNet::forward(const Tensor &input){
    Tensor A = input;

    for (auto& layer: layers){
        A = layer.forward(A);
    }

    return A;
}

/*
Calculates the gradient of MSE loss with respect to predictions

Operation: 2(predicted[i]-actual[i])/N
N = No of elements
Throws error if shapes differ
Outputs Tensor of same shape as input with loss gradients
*/
Tensor NeuralNet::dMSE(const Tensor& pred, const Tensor& target){
    if (pred.getShape() != target.getShape()){
        throw std::invalid_argument("Shape mismatch");
    }

    const std::vector<float>& p = pred.getData();
    const std::vector<float>& t = target.getData();

    std::vector<float> grad(p.size());

    for (int i = 0; i < grad.size(); i++){
        grad[i] = 2 * (p[i] - t[i]) / grad.size();
    }
    return Tensor(std::move(grad), pred.getShape());
}

/*
Runs inference for forward pass
Sequentially goes through all layers of NeuralNet 
Outputs Tensor with final predictions
*/
Tensor NeuralNet::predict(const Tensor& input){
    Tensor x = input;

    for (Layer& layer: layers){
        x = layer.forward(x);
    }

    return x;
}

/*
Trains NeuralNet using simple Schoastic Gradient Descent
Throws error if datasize mismatch
*/
void NeuralNet::Train(const std::vector<Tensor> &xTrain, const std::vector<Tensor> &yTrain, int epochs, float learnRate){

    if (xTrain.size() != yTrain.size()){
        throw std::invalid_argument("Dataset mismatch");
    }

    for (int e = 0; e < epochs; e++){
        for (size_t i = 0; i < xTrain.size(); i++){

            Tensor x = xTrain[i];

            for (Layer& layer : layers){
                x = layer.forward(x);
            }

            Tensor prediction = x;

            Tensor grad = dMSE(prediction, yTrain[i]);

            for (int l = (int)layers.size() - 1; l >= 0; l--){
                grad = layers[l].back(grad, learnRate);
            }
        }
    }
}

//* LAYER CLASS


/*
Initialises fully connected layer for NN.
Loads Random Weights and (Initialised) Zero Biases 
*/
Layer::Layer(int inputSize, int outputSize)
    : Weights(Tensor::RandomTensor({inputSize, outputSize}, -0.1f, 0.1f)),
      Bias(Tensor::ZeroTensor({1, outputSize}))
{}

/*
Forward Propagation for one layer.
inputCache: Input for backprop 
outputCache: Pre-Activation Input
*/
Tensor Layer::forward(const Tensor& input){
    inputCache = input;

    Tensor Z = (inputCache*Weights) + Bias;
    outputCache = Z;

    return Z.Relu();    
}

/*
Performs backpropagation through layers and updates parameters
gradOutput: Gradient from next layer
learnRate: Learning Rate
*/
Tensor Layer::back(const Tensor& gradOutput, float learnRate){
    Tensor dZ = gradOutput;
    dZ = dZ * (outputCache.dRelu());

    Tensor inputT = inputCache.transpose({1,0});
    dWeights = inputT.matmul(dZ);

    dBias = dZ.Rowsum();

    Tensor weightsT = Weights.transpose({1,0});
    Tensor gradInput = dZ.matmul(weightsT);

    Weights = Weights - (dWeights * learnRate);
    Bias = Bias - (dBias * learnRate);

    return gradInput;   
}