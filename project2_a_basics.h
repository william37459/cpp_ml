/// Virtual base classes for use in project, defining the required interfaces.
/// Also some helper functions that may be useful

// C++ includes
#include <cmath>
#include <cassert>

#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// For pair
#include <utility>

// Basic linear algebra provided
#include "dense_linear_algebra.h"

using namespace BasicDenseLinearAlgebra;

//=================================================================
/// Random number helper namespace
//=================================================================
namespace RandomNumber
{

    /// A "random device" that generates a new random number
    /// each time the program is run
    std::random_device Random_device;

    // Pseudo-random number generator seeded with
    // a random number
    // std::mt19937 Random_number_generator(Random_device());

    //  Alternative:

    /// Pseudo random number generator seeded with a
    /// constant integer (here 12345). This is useful for
    /// debugging because the program then produces the same sequence
    /// of random numbers each time it runs.
    std::mt19937 Random_number_generator(12345);

}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

//=================================================================
/// Base class for activation function
//=================================================================
class ActivationFunction
{

public:
    /// Empty constructor
    ActivationFunction() {};

    /// Name of the activation function; just some identifier so we can check
    /// if we've assigned the right one when reading in data for a trained
    /// network from a file. Makes sense to use the name of the class
    /// (which, sadly, can't be retrieved from within the code because
    /// different compilers do different name-mangling).
    virtual std::string name() const = 0;

    /// Definition of the activation function; pure virtual
    virtual double sigma(const double &x) = 0;

    /// Derivative of activation function; default implementation
    /// using FD
    virtual double dsigma(const double &x)
    {
        // Reference value
        double xx = x;
        double sigma_ref = sigma(xx);

        // Advanced value
        double fd_step = 1.0e-8;
        xx += fd_step;
        double sigma_pls = sigma(xx);

        // Return forward difference approximation
        return (sigma_pls - sigma_ref) / fd_step;
    }
};

//=================================================================
/// Tanh activation function
//=================================================================
class TanhActivationFunction : public ActivationFunction
{

public:
    /// Empty constructor
    TanhActivationFunction() {};

    /// Name of the function (so we can check if we've assigned
    /// the right one when reading in a trained network from
    /// a file)
    std::string name() const
    {
        return "TanhActivationFunction";
    }

    /// Definition of the activation function
    double sigma(const double &x)
    {
        return std::tanh(x);
    }

    /// Derivative of sigma function
    virtual double dsigma(const double &x)
    {
        return 1.0 / std::cosh(x) / std::cosh(x);
    }
};


/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////


//=================================================================
/// Base class for neural network; derive your own network from
/// this!
//=================================================================
class NeuralNetworkBasis
{

public:
    /// Evaluate the feed-forward algorithm, running through the entire network,
    /// for input x. Feed it through the non-input-layers and return the
    /// output from the final layer.
    virtual void feed_forward(const DoubleVector &input,
                              DoubleVector &output) const = 0;

    /// Get cost for given input and specified target output, so we're doing
    /// a single run through the network and compare the output to the target
    /// output.
    virtual double cost(const DoubleVector &input,
                        const DoubleVector &target_output) const = 0;

    /// Get cost for training data. The vector contains the training data
    /// in the form of pairs comprising
    ///   training_data[i].first  = input (a DoubleVector)
    ///   training_data[i].second = target_output (a DoubleVector)
    virtual double cost_for_training_data(
        const std::vector<std::pair<DoubleVector, DoubleVector>> training_data)
        const = 0;

    /// Write parameters for network to file (e.g. following
    /// successful training). Format:
    ///
    /// For each non-input layer:
    ///
    ///    name of activation function (literal string)
    ///    number of neurons in previuos layer (unsigned)
    ///    number of neurons in current layer
    ///    0 b[0] (index and entry in bias vector)
    ///     :
    ///    n_neuron-1 b[n_neuron-1]
    ///    0 0 a[0][0] (indices and entry in weight matrix)
    ///    0 1 a[0][1]
    ///     :
    ///    1 0 a[1][0]
    ///    1 1 a[1][1]
    ///     :
    virtual void write_parameters_to_disk(const std::string &filename) const = 0;

    /// Read in parameters for network (e.g. from previously
    /// performed training). Ssee write_parameters_to_disk(...) for format
    virtual void read_parameters_from_disk(const std::string &filename) = 0;

    /// Train the network: specify filename containing training data
    /// in the form
    ///
    ///     training_data[i].first  = input
    ///     training_data[i].second = target output
    ///
    /// as well as the learning rate, the convergence tolerance and
    /// the max. number of iterations. Optional final argument: filename
    /// of the file used to document the convergence history.
    /// No convergence history is written if string is empty or not provided.
    virtual void train(
        const std::vector<std::pair<DoubleVector, DoubleVector>> &training_data,
        const double &learning_rate,
        const double &tol_training,
        const unsigned &max_iter,
        const std::string &convergence_history_file_name = "") = 0;

    /// Initialise parameters (weights and biases), drawing random numbers from
    /// a normal distribution with specified mean and standard deviation.
    /// This function is broken but demonstrates how to draw normally
    /// distributed random numbers. Please reimplement it so that the random
    /// values are assigned to your weights and biases.
    virtual void initialise_parameters(const double &mean, const double &std_dev)
    {
        // Set up a normal distribution.

        // Calling "normal_dist(rnd)" then returns a random number drawn
        // from this distribution
        std::normal_distribution<> normal_dist(mean, std_dev);

        // TODO: This is a dummy loop; make it visit all the weights and
        //       biases and assign the random number from the normal distribution
        unsigned n_entries = 100;
        for (unsigned i = 0; i < n_entries; i++)
        {
            double weight_or_bias = normal_dist(RandomNumber::Random_number_generator);
        }

        throw std::runtime_error("Never get here! Please overload this function\n");
    }

    // Helper functions
    //------------------

    /// Read training data from disk. Format:
    ///
    /// n_training_sets
    /// n_input
    /// n_output
    /// x[1], x[2],... x[n_input] y[1], y[2],... y[n_output]
    ///                    :
    ///     n_training_sets lines of data in total
    ///                    :
    /// x[1], x[2],... x[n_input] y[1], y[2],... y[n_output]
    /// end_of_file (literal string)
    ///
    virtual void read_training_data(
        const std::string &filename,
        std::vector<std::pair<DoubleVector, DoubleVector>> &training_data)
        const
    {
        // Wipe training data so we can push back
        training_data.clear();

        // Read parameters
        std::ifstream training_data_file(filename.c_str());
        unsigned n_training_sets = 0;
        training_data_file >> n_training_sets;
        unsigned n_input = 0;
        training_data_file >> n_input;
        unsigned n_output = 0;
        training_data_file >> n_output;

        // Read the actual data
        DoubleVector input(n_input);
        DoubleVector output(n_output);
        for (unsigned i = 0; i < n_training_sets; i++)
        {
            for (unsigned j = 0; j < n_input; j++)
            {
                training_data_file >> input[j];
            }
            for (unsigned j = 0; j < n_output; j++)
            {
                training_data_file >> output[j];
            }
            training_data.push_back(std::make_pair(input, output));
        }

        // Have we reached the end?
        std::string test_string;
        training_data_file >> test_string;
        if (test_string != "end_of_file")
        {
            throw std::runtime_error("\n\nERROR: Training set data doesn't end with \"end_of_file\"\n\n");
        }
        else
        {
            std::cout << "Yay! Succesfully read training data in\n\n"
                      << "        " << filename
                      << "\n"
                      << std::endl;
        }
    }

    /// Helper function to output training data. Given N sets
    /// of n-dimensional input (i1,...,in) and m-dimensional target
    /// output, writes  (t1,...,tm) this function writes N lines
    /// containing
    ///  i1,i2,...,in, t1,t2,...,tm
    /// to the specified output stream.
    void output_training_data(
        std::ofstream &outfile,
        const std::vector<std::pair<DoubleVector, DoubleVector>> &training_data)
        const
    {
        unsigned n = training_data.size();
        for (unsigned i = 0; i < n; i++)
        {
            unsigned m = (training_data[i].first).n();
            for (unsigned j = 0; j < m; j++)
            {
                outfile << (training_data[i].first)[j] << " ";
            }
            m = (training_data[i].second).n();
            for (unsigned j = 0; j < m; j++)
            {
                outfile << (training_data[i].second)[j] << " ";
            }
            outfile << std::endl;
        }
    }

    /// Helper function to output training data: Given N sets
    /// of n-dimensional input (i1,...,in) and m-dimensional target
    /// output, writes  (t1,...,tm) this function writes N lines
    /// containing
    ///  i1,i2,...,in, t1,t2,...,tm
    /// to the specified output file.
    void output_training_data(
        const std::string &output_filename,
        const std::string &training_data_filename) const
    {
        // Read training data
        std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
        bool do_sanity_check = true;
        read_training_data(training_data_filename, training_data);

        // Do the actual output
        std::ofstream outfile(output_filename.c_str());
        NeuralNetworkBasis::output_training_data(outfile, training_data);
        outfile.close();
    }

    /// Output result from trained network for specified inputs
    void output(std::string filename,
                const std::vector<DoubleVector> &input) const
    {
        std::ofstream outfile(filename.c_str());
        output(outfile, input);
        outfile.close();
    }

    /// Output result from trained network for specified inputs
    void output(std::ofstream &outfile,
                const std::vector<DoubleVector> &input) const
    {
        unsigned npts = input.size();
        for (unsigned i = 0; i < npts; i++)
        {
            DoubleVector current_input(input[i]);
            unsigned n = current_input.n();
            for (unsigned j = 0; j < n; j++)
            {
                outfile << current_input[j] << " ";
            }

            // Feed through network; output gets resized automatically
            DoubleVector output;
            feed_forward(current_input, output);
            n = output.n();
            for (unsigned j = 0; j < n; j++)
            {
                outfile << output[j] << " ";
            }
            outfile << std::endl;
        }
    }
};
