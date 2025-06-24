#include "project2_a.h"

using namespace BasicDenseLinearAlgebra;

int main()
{
    //initializing the activation function
    ActivationFunction *activationFunction = new TanhActivationFunction();
    
    // Other Neural networks that were tested
    // std::vector<NeuralNetwork *>all_nns ={ 
    //     new NeuralNetwork(2, {
    //         {4, activationFunction},
    //         {4, activationFunction},
    //         {1, activationFunction},
    //     }),
    //     new NeuralNetwork(2, {
    //         {4, activationFunction},
    //         {4, activationFunction},
    //         {4, activationFunction},
    //         {1, activationFunction},
    //     }),
    //     new NeuralNetwork(2, {
    //         {4, activationFunction},
    //         {4, activationFunction},
    //         {4, activationFunction},
    //         {4, activationFunction},
    //         {1, activationFunction},
    //     }),
    // };

    // std::vector<NeuralNetwork *>all_nns ={ 
    //     new NeuralNetwork(2, {
    //         {4, activationFunction},
    //         {4, activationFunction},
    //         {1, activationFunction},
    //     }),
    //     new NeuralNetwork(2, {
    //         {8, activationFunction},
    //         {8, activationFunction},
    //         {1, activationFunction},
    //     }),
    //     new NeuralNetwork(2, {
    //         {16, activationFunction},
    //         {16, activationFunction},
    //         {1, activationFunction},
    //     }),
    // };

    // std::vector<double> computation_times;

    // Loop if you want to test the computation time of the neural network or try different neural networks
    // for(unsigned i = 0; i < 100; ++i){
        NeuralNetwork *nn = new NeuralNetwork(2, {
                {16, activationFunction},
                {16, activationFunction},
                {1, activationFunction},
            });

        std::__1::vector<std::__1::pair<BasicDenseLinearAlgebra::DoubleVector, BasicDenseLinearAlgebra::DoubleVector>> training_data;

        nn->read_training_data("project_training_data.dat", training_data);

        std::vector<DoubleVector> input_data;

        // Extract the first values
        for (const std::__1::pair<BasicDenseLinearAlgebra::DoubleVector, BasicDenseLinearAlgebra::DoubleVector>& input : training_data) {
            input_data.push_back(input.first);
        }
        
        // To compute the time it takes to train the neural network
        auto start = std::chrono::high_resolution_clock::now();

        nn->train(training_data, 0.01, 1e-2, 1e6, "convergence_history.dat");

        // To compute the time it takes to train the neural network
        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //     computation_times.push_back(duration.count());
    // }
    return 0;
}