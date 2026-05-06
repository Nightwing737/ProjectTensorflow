// #include "DataProcessing.h"

// std::array<std::vector<float>,4> Data::TrainTestSplit(Tensor x, Tensor y, float testSplit){
//     int xSize = x.getSize();
//     int ySize = y.getSize();

//     std::random_device rd;
//     std::mt19937 g(rd());
    
//     int trainSplit = 1 - testSplit;

//     std::vector<float> xShuffle = x.getData();
//     std::vector<float> yShuffle = y.getData();

//     std::shuffle(xShuffle.begin(),xShuffle.end(),g);
//     std::shuffle(yShuffle.begin(),yShuffle.end(),g);

//     int xtrainIndex = round(xSize*trainSplit); 
//     int ytrainIndex = round(ySize*trainSplit);

//     std::vector<float> xTrain(xShuffle.begin(), xShuffle.begin() + xtrainIndex);
//     std::vector<float> xTest(xShuffle.begin() + xtrainIndex, xShuffle.end());
//     std::vector<float> yTrain(yShuffle.begin(), yShuffle.begin() + ytrainIndex);
//     std::vector<float> yTest(yShuffle.begin() + ytrainIndex, yShuffle.end());
//     return {std::move(xTrain), std::move(xTest), std::move(yTrain), std::move(yTest)};
// }



