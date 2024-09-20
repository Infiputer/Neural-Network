#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

enum ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    IDENTITY
};

class NeuralNetwork
{
public:
    // Constructor to initialize network sizes and weights
    NeuralNetwork(size_t input_len, const std::vector<size_t> &hidden_lens, size_t output_len, bool use_bias);
    NeuralNetwork();


    void initializeWeightsAndBiases();
    void forward(const std::vector<float> &neural_input);
    void learn(const std::vector<float> &neural_input, const std::vector<float> &target, float learning_rate);
    uint8_t save(const std::string &file);
    uint8_t load(const std::string &file);

    // Define the activation functions and its derivatives
    inline float relu(float x);
    inline float relu_derivative(float x);
    inline float sigmoid(float x);
    inline float sigmoid_derivative(float x);
    inline float tanh(float x);
    inline float tanh_derivative(float x);
    inline float identity(float x);
    inline float identity_derivative(float x);

    float activate(float x);
    float activateDerivative(float x);

    ActivationFunction activationFunction;
    void setActivationFunction(ActivationFunction func);

    // Define the softmax function
    std::vector<float> softmax(const std::vector<float> &x);

    // Compute the hidden layer activations
    void computeHiddenLayer(const std::vector<float> &input, size_t layer_idx);

    // Compute the output layer activations
    void computeOutputLayer();

    size_t inp_len;                          // Number of input neurons
    std::vector<size_t> hid_lens;            // Sizes of hidden layers
    size_t out_len;                          // Number of output neurons
    std::vector<std::vector<float>> weights; // Weights for connections
    std::vector<std::vector<float>> biases;  // Biases for neurons

    std::vector<std::vector<float>> hidden; // Hidden layers
    std::vector<float> output;              // Output layer
    std::vector<float> output_error;        // Output error for backpropagation

    bool use_biases;
};