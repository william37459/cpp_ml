#include "dense_linear_algebra.h"
#include "project2_a_basics.h"

#include <algorithm>

using namespace BasicDenseLinearAlgebra;

class NeuralNetworkLayer
{
public:
    unsigned numberOfNeurons;
    ActivationFunction *activationFunction;
    DoubleMatrix weights;
    DoubleVector biases;

    NeuralNetworkLayer(int numNeurons, int inputSize, ActivationFunction *activation)
        : numberOfNeurons(numNeurons), activationFunction(activation), weights(numNeurons, inputSize)
    {
        biases.resize(numberOfNeurons);
    }

    void setWeights(const DoubleMatrix &newWeights)
    {
        if (newWeights.n() != weights.n() || newWeights.m() != weights.m())
        {
            throw std::invalid_argument("New weights matrix must have the same dimensions as the current weights matrix.");
        }

        weights = newWeights;
    }

    void setBias(const DoubleVector &newBiases)
    {
        if (newBiases.n() != biases.n())
        {
            throw std::invalid_argument("New bias vector must have the same size as the current bias vector.");
        }

        biases = newBiases;
    }

    DoubleVector processInput(const DoubleVector &input) const
    {
        if (input.n() != weights.m())
        {
            throw std::invalid_argument("Input size does not match the number of inputs expected by the layer.");
        }

        DoubleVector output(numberOfNeurons);

        // Computing sigma(W*x + b) for each neuron
        for (int i = 0; i < numberOfNeurons; ++i)
        {
            double sum = biases[i];
            for (size_t j = 0; j < input.n(); ++j)
            {
                sum += weights(i, j) * input[j];
            }
            output[i] = activationFunction->sigma(sum);
        }

        return output;
    }
};

class NeuralNetwork : public NeuralNetworkBasis
{
public:
    NeuralNetwork(
        const unsigned &n_input,
        const std ::vector<std::pair<unsigned, ActivationFunction *>> &
            non_input_layer)
    {
        // Initialize the first hidden layer
        layers.push_back(NeuralNetworkLayer(non_input_layer[0].first, n_input, non_input_layer[0].second));

        // Initialize the non-input layers
        for (unsigned index = 1; index < non_input_layer.size(); index++)
        {
            layers.push_back(NeuralNetworkLayer(non_input_layer[index].first, non_input_layer[index - 1].first, non_input_layer[index].second));
        }
    }

    virtual void initialise_parameters(const double &mean, const double &std_dev) override
    {
        for (auto &layer : layers)
        {
            // Xavier He initialization for a better convergence
            double xavier_he_std_dev = std::sqrt(2.0 / (layer.weights.m() + layer.weights.n()));
            // Random number generator
            std::normal_distribution<> normal_dist(mean, xavier_he_std_dev);       

            // Initialize weights
            for (unsigned i = 0; i < layer.weights.n(); ++i)
            {
                for (unsigned j = 0; j < layer.weights.m(); ++j)
                {
                    layer.weights(i, j) = normal_dist(RandomNumber::Random_number_generator);
                }
            }

            // Initialize biases
            layer.biases.resize(layer.biases.n());

        }
    }

    virtual void feed_forward(const DoubleVector &input, DoubleVector &output) const override
    {
        if (layers.empty())
        {
            throw std::runtime_error("The network has no layers to process input.");
        }

        // We start with the initial input
        DoubleVector current_input = input; 

        // Propagate through each layer
        for (const auto &layer : layers)
        {
            current_input = layer.processInput(current_input); // Process the input through the layer
        }

        // Set the output value from the last layer
        output = current_input;
    }

    virtual double cost(const DoubleVector &input, const DoubleVector &target_output) const override
    {
        // Feed-forward to get the predicted output (a^[L](input))
        DoubleVector predicted_output;
        feed_forward(input, predicted_output);

        // Check that the predicted output and target output have the same size
        if (predicted_output.n() != target_output.n())
        {
            throw std::invalid_argument("Predicted output and target output must have the same size.");
        }

        // Compute the L^2 error for this single example (sum of squared differences)
        double squared_error = 0.0;
        for (size_t i = 0; i < predicted_output.n(); ++i)
        {
            double diff = predicted_output[i] - target_output[i];
            squared_error += diff * diff; // (y - a)^2
        }

        // Apply the 1/2 factor to match the formula
        double cost_value = 0.5 * squared_error; // 1/2 * sum of squared errors

        return cost_value;
    }

    virtual double cost_for_training_data(const std::vector<std::pair<DoubleVector, DoubleVector>> training_data) const override
    {
        double total_cost = 0.0;
        unsigned N = training_data.size();

        // Loop over all training examples
        for (const auto &data : training_data)
        {
            // Training data 
            const DoubleVector &input = data.first; 
            // Targeted output
            const DoubleVector &target_output = data.second;

            // Compute the cost for this example and add it to the total cost
            total_cost += cost(input, target_output);
        }

        // Return the average cost over all examples
        total_cost /= N;
        return total_cost;
    }

    virtual void read_parameters_from_disk(const std::string &filename) override
    {
        // Open the file 
        std::ifstream file(filename);

        // Check if the file was opened successfully
        if (!file.is_open())
            throw std::runtime_error("Could not open file: " + filename);


        // Variables to store the data from the file
        std::string line;
        std::string activationFunctionName;
        unsigned layerIndex = 0;

        // Read the parameters for each layer line by line
        while (file >> activationFunctionName)
        {
            // Check if the activation functon is valid
            if (activationFunctionName != layers[layerIndex].activationFunction->name())
                throw std::runtime_error("Activation function not recognized: " + activationFunctionName);

            // Get the dimensions of the layer
            unsigned m;
            unsigned n;
            file >> m;
            file >> n;

            // Gets bias vector
            DoubleVector biases(n);
            biases.read(file);

            // Gets weight matrix
            DoubleMatrix weights(n, m);
            weights.read(file);

            // Check if the dimensions of the layer match the network
            if(layerIndex >= layers.size())
                throw std::runtime_error("Number of layers in file does not match the number of layers in the network.");

            // Check if the dimensions of the layer match the network
            if(layers[layerIndex].numberOfNeurons != n)
                throw std::runtime_error("Number of neurons in file does not match the number of neurons in the network.");

            // Set the weights and biases for the layer
            layers[layerIndex].setWeights(weights);
            layers[layerIndex].setBias(biases);
            ++layerIndex;
        }

        // Check if the number of layers in the file match the number of layers in the network
        if(layerIndex != layers.size()){
            throw std::runtime_error("Number of layers in file does not match the number of layers in the network.");
        }
        file.close();
    }

    virtual void write_parameters_to_disk(const std::string &filename) const override
    {

        // Open the file for writing
        std::ofstream outfile(filename);
        if (!outfile.is_open())
        {
            throw std::runtime_error("Unable to open file for writing: " + filename);
        }

        // Write each non-input layer's parameters
        for (unsigned index = 0; index < layers.size(); ++index)
        {
            // Write the activation function name
            outfile << layers[index].activationFunction->name() << std::endl;

            // Write the input dimension (previous layer's number of neurons)
            unsigned m = layers[index].weights.m();
            outfile << m << std::endl;

            // Write the number of neurons in the current layer
            unsigned n = layers[index].numberOfNeurons;
            outfile << n << std::endl;

            // Write the bias vector: one line per bias, format: j bj
            layers[index].biases.output(outfile);

            // Write the weight matrix: format: i j aij
            layers[index].weights.output(outfile);
        }

        outfile.close();
    }

    void train(
        const std::vector<std::pair<DoubleVector, DoubleVector>> &training_data,
        const double &learning_rate,
        const double &tol_training,
        const unsigned &max_iter,
        const std::string &convergence_history_file_name) override
    {
        // Initialize the parameters with a normal distribution
        initialise_parameters(0.0, 0.005);

        // Variables for training
        unsigned iteration = 0;
        double current_cost = 0.0;
        
        // Open file for logging convergence history
        std::ofstream convergence_history_file;

        // Check if a filename was provided
        if (!convergence_history_file_name.empty())
        {
            convergence_history_file.open(convergence_history_file_name);

            // Check if the file was opened successfully
            if (!convergence_history_file.is_open())
            {
                throw std::runtime_error("Unable to open file for convergence history: " + convergence_history_file_name);
            }
        }

        // Random number generator to select training point
        std::random_device rd;                                            
        std::mt19937 gen(rd());                                           
        std::uniform_int_distribution<> dis(0, training_data.size() - 1); 

        // Training loop
        while (iteration < max_iter)
        {
            // Select a random training point
            std::pair<DoubleVector, DoubleVector> training_point = training_data[dis(gen)];

            // Apply the desired algorithm to compute the gradients and update the weights and biases
            backpropagation(training_point, learning_rate);
            //finite_difference(training_point, learning_rate);

            // Compute the cost every 1000 iterations and log the total_cost
            if(iteration % 1000 == 0) {
                
                //Compute the total cost
                current_cost = cost_for_training_data(training_data);
                
                // Log the total cost
                convergence_history_file << iteration << " " << current_cost << std::endl;
                
                // Check if the cost is acceptable
                if (std::abs(current_cost) < tol_training)
                {
                    break;
                }
            }
            ++iteration;
        }

        // Close convergence history file if it was opened
        if (convergence_history_file.is_open())
        {
            convergence_history_file.close();
        }

        // Notify if maximum iterations were reached
        if (iteration == max_iter)
        {
            std::cout << "Training finished after reaching the maximum number of iterations (" << max_iter << ")." << std::endl;
        }
    }

private:
    std::vector<DoubleMatrix> layers_weights;
    std::vector<DoubleVector> layers_biases;
    std::vector<NeuralNetworkLayer> layers;

    void finite_difference(std::pair<DoubleVector, DoubleVector> &training_point, double learning_rate)
    {
        // Compute the initial cost for the chosen training example
        DoubleVector input = training_point.first;
        DoubleVector target_output = training_point.second;
        double cost_initial = cost(input, target_output);

        // Create structures to store gradients
        std::vector<DoubleMatrix> weight_gradients;
        std::vector<DoubleVector> bias_gradients;

        // Compute the gradient of the cost function using finite differences
        double epsilon = 1e-4;
        for (NeuralNetworkLayer &layer : layers)
        {
            // Initialize gradient storage by creating objects with required dimensions
            DoubleMatrix weight_grad(layer.weights.n(), layer.weights.m());
            DoubleVector bias_grad(layer.biases.n());

            // Gradient for weights
            for (unsigned i = 0; i < layer.weights.n(); ++i)
            {
                for (unsigned j = 0; j < layer.weights.m(); ++j)
                {
                    // Perturbate the weight
                    layer.weights(i, j) += epsilon;

                    // Compute the cost with the perturbed weight
                    double cost_offset = cost(input, target_output);

                    // Reset the weight to its original value
                    layer.weights(i, j) -= epsilon;

                    // Compute the gradient estimate
                    double gradient = (cost_offset - cost_initial) / epsilon;

                    // Store the gradient
                    weight_grad(i, j) = gradient;
                }
            }

            // Gradient for biases
            for (unsigned i = 0; i < layer.biases.n(); ++i)
            {
                // Perturbate the bias
                layer.biases[i] += epsilon;

                // Compute the cost with the perturbed bias
                double cost_offset = cost(input, target_output);

                // Reset the bias to its original value
                layer.biases[i] -= epsilon;

                // Compute the gradient estimate
                double gradient = (cost_offset - cost_initial) / epsilon;

                // Store the gradient
                bias_grad[i] = gradient;
            }

            // Add gradients to the vectors
            weight_gradients.push_back(weight_grad);
            bias_gradients.push_back(bias_grad);
        }

        // Apply the updates after computing all gradients
        for (unsigned index = 0; index < layers.size(); ++index)
        {
            auto &layer = layers[index];
            DoubleMatrix &weight_grad = weight_gradients[index];
            DoubleVector &bias_grad = bias_gradients[index];

            // Update weights
            for (unsigned i = 0; i < layer.weights.n(); ++i)
            {
                for (unsigned j = 0; j < layer.weights.m(); ++j)
                {
                    layer.weights(i, j) -= learning_rate * weight_grad(i, j);
                }
            }

            // Update biases
            for (unsigned i = 0; i < layer.biases.n(); ++i)
            {
                layer.biases[i] -= learning_rate * bias_grad[i];
            }
        }
    }

    void backpropagation(std::pair<DoubleVector, DoubleVector> &training_point, double learning_rate)
    {
        std::vector<DoubleVector> activations;
        std::vector<DoubleVector> zs;
        activations.push_back(training_point.first);

        // Forward pass
        for (unsigned index = 0; index < layers.size(); ++index)
        {
            // Initialize the z and a vectors for the current layer
            DoubleVector z(layers[index].numberOfNeurons);
            DoubleVector a(layers[index].numberOfNeurons);
            for (unsigned i = 0; i < layers[index].weights.n(); ++i)
            {
                double sum = 0.0;
                for (unsigned j = 0; j < activations.back().n(); ++j)
                {
                    // Compute the weighted sum of the inputs
                    sum += layers[index].weights(i, j) * activations.back()[j];
                }
                // Populate the z and a vectors
                z[i] = sum + layers[index].biases[i];;
                a[i] = layers[index].activationFunction->sigma(z[i]);
            }
            zs.push_back(z);
            activations.push_back(a);
        }

        // Backward pass
        std::vector<DoubleVector> all_delta(layers.size());
        DoubleVector delta(activations.back().n());
        // Compute the error for the output layer
        for (unsigned i = 0; i < activations.back().n(); ++i)
        {
            double d_cost = activations.back()[i] - training_point.second[i];
            delta[i] = d_cost * layers.back().activationFunction->dsigma(zs.back()[i]);
        }
        all_delta.back() = delta;

        // Compute the error for the hidden layers
        for (int l = layers.size() - 2; l >= 0; --l)
        {
            DoubleVector delta(layers[l].numberOfNeurons);
            for (unsigned i = 0; i < layers[l].numberOfNeurons; ++i)
            {
                double sum = 0.0;
                
                //Calculating the error indices
                for (unsigned j = 0; j < layers[l + 1].numberOfNeurons; ++j)
                {
                    sum += layers[l + 1].weights(j, i) * all_delta[l + 1][j];
                }
                delta[i] = sum * layers[l].activationFunction->dsigma(zs[l][i]);
                 
                //  // Update weights for the current neuron
                // for (unsigned j = 0; j < layers[l].weights.m(); ++j)
                // {
                //     layers[l].weights(i, j) -= learning_rate * delta[i] * activations[l][j];
                // }
                
                // // Update biases for the current neuron
                // layers[l].biases[i] -= learning_rate * delta[i];
            }

            // Store the delta for the current layer
            all_delta[l] = delta;
        }

        // Old propogation implementation
        // Update weights and biases
        for (unsigned l = 0; l < layers.size(); ++l)
        {
            for (unsigned i = 0; i < layers[l].weights.n(); ++i)
            {
                for (unsigned j = 0; j < layers[l].weights.m(); ++j)
                {
                    layers[l].weights(i, j) -= learning_rate * all_delta[l][i] * activations[l][j];
                }
                layers[l].biases[i] -= learning_rate * all_delta[l][i];
            }
        }
    }
};
