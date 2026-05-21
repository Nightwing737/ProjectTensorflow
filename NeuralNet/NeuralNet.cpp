#include "NeuralNet.h"

NeuralNet::NeuralNet(const std::vector<int>& topology){
    for (int i = 0; i < topology.size()-1; i++){
        int inputSize = topology[i];
        int outputSize = topology[i+1];

        layers.emplace_back(inputSize, outputSize);
    }
}

Tensor NeuralNet::forward(const Tensor &input){
    Tensor A = input;

    for (auto& layer: layers){
        A = layer.forward(A);
    }

    return A;
}

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





Layer::Layer(int inputSize, int outputSize)
    : Weights(Tensor::RandomTensor({inputSize, outputSize}, -0.1f, 0.1f)),
      Bias(Tensor::ZeroTensor({1, outputSize}))
{
}


Tensor Layer::forward(const Tensor& input){
    inputCache = input;

    Tensor Z = (inputCache*Weights) + Bias;
    outputCache = Z;

    return Z.Relu();    
}

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

Tensor NeuralNet::predict(const Tensor& input){
    Tensor x = input;

    for (Layer& layer: layers){
        x = layer.forward(x);
    }

    return x;
}

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