# ProjectTensorflow

A zero-dependency, pure C++ Machine Learning and Linear Algebra library.

## Tech Stack And Architecture
This project is built entirely from scratch with zero external dependencies. It relies only on the standard C++ library.

Core Components:

* Custom Tensor Engine: A bespoke Tensor class handling foundational linear algebra and matrix operations.

* Data Processing Pipeline: Custom DataProcessing classes for raw data ingestion and mutation.

* ML Implementations: From-scratch algorithms (NN and LinReg) built exclusively on top of the custom Tensor and Data APIs.

## Quickstart
There are no dependencies, frameworks, or package managers required. You only need a standard C++ compiler (e.g., GCC or Clang).

1. Clone the Repo
    ```bash
    git clone https://github.com/Nightwing737/MiniML.git
    cd MiniML
    ```

2. Compile the Source

    Example compilation using g++. Adjust if you are using a Makefile or CMake.
    ```bash
    g++ -std=c++17 -O3 -I./Tensors -I./DataProcess src/main.cpp -o ml_project
    ```

3. Run the demo
    ```bash
    ./ml_project
    ```

## Project Structure
The architecture strictly separates the foundational mathematics from the model implementations. NN and LinReg are entirely dependent on the core Tensors and DataProcess bases.

```
├── DataProcessing/    # Base: Custom data loading and preprocessing pipelines
├── Tensors/        # Base: Custom C++ Tensor implementation (core math)
├── LinearRegression/         # Dependent: Linear Regression models 
├── NeuralNet/             # Dependent: Neural Network implementations
├── TensorDocs.md         # Tensor API Reference
├── NN_Data_Docs.md       # NeuralNet and DataProcessing API Reference
├── src/                  # Contains main.cpp where code goes
└── README.md       # You are here
```

## Documentation
This README is only for getting the code running. For the full C++ API reference, memory management details, and explanations of operations, read the respective Documentations (TensorDocs.md and NN_Data_Docs.md).

---