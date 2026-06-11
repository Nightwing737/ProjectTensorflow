#include "DataProcessing.h"
/*
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
*/
std::array<std::vector<float>,4> Data::TrainTestSplit(Tensor x, Tensor y, float testSplit){
    int xSize = x.getSize();
    int ySize = y.getSize();

    std::random_device rd;
    std::mt19937 g(rd());
    
    int trainSplit = 1 - testSplit;

    std::vector<float> xShuffle = x.getData();
    std::vector<float> yShuffle = y.getData();

    std::shuffle(xShuffle.begin(),xShuffle.end(),g);
    std::shuffle(yShuffle.begin(),yShuffle.end(),g);

    int xtrainIndex = round(xSize*trainSplit); 
    int ytrainIndex = round(ySize*trainSplit);

    std::vector<float> xTrain(xShuffle.begin(), xShuffle.begin() + xtrainIndex);
    std::vector<float> xTest(xShuffle.begin() + xtrainIndex, xShuffle.end());
    std::vector<float> yTrain(yShuffle.begin(), yShuffle.begin() + ytrainIndex);
    std::vector<float> yTest(yShuffle.begin() + ytrainIndex, yShuffle.end());
    return {std::move(xTrain), std::move(xTest), std::move(yTrain), std::move(yTest)};
}



