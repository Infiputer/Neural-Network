#include "neural.h"

void NeuralNetwork::setActivationFunction(ActivationFunction func)
{
    activationFunction = func;
}

float NeuralNetwork::activate(float x)
{
    switch (activationFunction)
    {
    case RELU:
        return relu(x);
    case SIGMOID:
        return sigmoid(x);
    case TANH:
        return tanh(x);
    case IDENTITY:
        return identity(x);
    // Add more cases as needed
    default:
        return relu(x); // Default to ReLU if undefined
    }
}

float NeuralNetwork::activateDerivative(float x)
{
    switch (activationFunction)
    {
    case RELU:
        return relu_derivative(x);
    case SIGMOID:
        return sigmoid_derivative(x);
    case TANH:
        return tanh_derivative(x);
    case IDENTITY:
        return identity_derivative(x);
    // Add more cases as needed
    default:
        return relu_derivative(x); // Default to ReLU derivative if undefined
    }
}

// Define the ReLU activation function
inline float NeuralNetwork::relu(float x)
{
    return (x > 0) ? x : 0.0f;
}

inline float NeuralNetwork::relu_derivative(float x)
{
    return (x > 0) ? 1.0f : 0.0f;
}

inline float NeuralNetwork::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float NeuralNetwork::sigmoid_derivative(float x)
{
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

inline float NeuralNetwork::tanh(float x)
{
    return std::tanh(x);
}

inline float NeuralNetwork::tanh_derivative(float x)
{
    float t = tanh(x);
    return 1 - t * t;
}

inline float NeuralNetwork::identity(float x)
{
    return x;
}
inline float NeuralNetwork::identity_derivative(float x)
{
    return 1;
}

// Define the softmax function
std::vector<float> NeuralNetwork::softmax(const std::vector<float> &x)
{
    std::vector<float> result(x.size());
    float max_elem = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;

    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = std::exp(x[i] - max_elem);
        sum += result[i];
    }

    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] /= sum;
    }

    return result;
}

// Constructor to initialize network sizes and weights
NeuralNetwork::NeuralNetwork(size_t input_len, const std::vector<size_t> &hidden_lens, size_t output_len, bool use_bias)
    : inp_len(input_len), hid_lens(hidden_lens), out_len(output_len), use_biases(use_bias), activationFunction(RELU)
{
    // Initialize weights and biases
    size_t prev_len = inp_len;
    for (size_t len : hid_lens)
    {
        weights.push_back(std::vector<float>(prev_len * len));
        if (use_biases)
            biases.push_back(std::vector<float>(len));
        prev_len = len;
    }
    weights.push_back(std::vector<float>(prev_len * out_len));
    if (use_biases)
        biases.push_back(std::vector<float>(out_len));

    hidden.resize(hid_lens.size());
    for (size_t i = 0; i < hid_lens.size(); ++i)
    {
        hidden[i].resize(hid_lens[i]);
    }
    output.resize(out_len);
    output_error.resize(out_len);
}

NeuralNetwork::NeuralNetwork()
    : inp_len(0), out_len(0), use_biases(true), activationFunction(RELU)
{
    hid_lens.clear();
    weights.clear();
    biases.clear();
    hidden.clear();
    output.clear();
    output_error.clear();
}

void NeuralNetwork::initializeWeightsAndBiases()
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 0.1); // He initialization

    for (auto &layer_weights : weights)
        for (auto &weight : layer_weights)
            weight = distribution(generator);
    if (use_biases)
    {
        for (auto &layer_biases : biases)
            for (auto &bias : layer_biases)
                bias = distribution(generator);
    }
}

// Compute the hidden layer activations
void NeuralNetwork::computeHiddenLayer(const std::vector<float> &input, size_t layer_idx)
{
    const std::vector<float> &prev_layer = (layer_idx == 0) ? input : hidden[layer_idx - 1];
    const std::vector<float> &layer_weights = weights[layer_idx];
    std::vector<float> &layer_biases = biases[layer_idx];
    std::vector<float> &layer_output = hidden[layer_idx];

    size_t prev_len = (layer_idx == 0) ? inp_len : hid_lens[layer_idx - 1];
    size_t curr_len = hid_lens[layer_idx];

    for (size_t x = 0; x < curr_len; ++x)
    {
        float sum = (use_biases) ? layer_biases[x] : 0.0f;
        for (size_t i = 0; i < prev_len; ++i)
        {
            sum += prev_layer[i] * layer_weights[i * curr_len + x];
        }
        layer_output[x] = activate(sum);
    }
}

void NeuralNetwork::computeOutputLayer()
{
    const std::vector<float> &last_hidden = hidden.back();
    const std::vector<float> &layer_weights = weights.back();
    std::vector<float> &layer_biases = biases.back();

    size_t prev_len = hid_lens.back();
    size_t curr_len = out_len;

    std::vector<float> pre_activation(curr_len);

    for (size_t x = 0; x < curr_len; ++x)
    {
        float sum = (use_biases) ? layer_biases[x] : 0.0f;
        for (size_t i = 0; i < prev_len; ++i)
        {
            sum += last_hidden[i] * layer_weights[i * curr_len + x];
        }
        pre_activation[x] = sum;
    }

    output = softmax(pre_activation);
}

void NeuralNetwork::forward(const std::vector<float> &neural_input)
{
    for (size_t i = 0; i < hid_lens.size(); ++i)
    {
        computeHiddenLayer(neural_input, i);
    }
    computeOutputLayer();
}

void NeuralNetwork::learn(const std::vector<float> &neural_input, const std::vector<float> &target, float learning_rate)
{
    forward(neural_input);

    // Calculate output error and delta
    for (size_t i = 0; i < out_len; ++i)
    {
        output_error[i] = output[i] - target[i];
    }

    // Backpropagation for hidden layers
    std::vector<std::vector<float>> hidden_errors(hid_lens.size());
    for (size_t i = hid_lens.size(); i-- > 0;)
    {
        hidden_errors[i].resize(hid_lens[i]);

        for (size_t j = 0; j < hid_lens[i]; ++j)
        {
            float error = 0.0f;
            if (i == hid_lens.size() - 1) // Last hidden layer
            {
                for (size_t k = 0; k < out_len; ++k)
                {
                    error += output_error[k] * weights[i + 1][j * out_len + k];
                }
            }
            else // Other hidden layers
            {
                for (size_t k = 0; k < hid_lens[i + 1]; ++k)
                {
                    error += hidden_errors[i + 1][k] * weights[i + 1][j * hid_lens[i + 1] + k];
                }
            }
            hidden_errors[i][j] = error * activateDerivative(hidden[i][j]);
        }
    }

    // Update weights and biases for output layer
    for (size_t i = 0; i < out_len; ++i)
    {
        for (size_t j = 0; j < hid_lens.back(); ++j)
        {
            weights.back()[j * out_len + i] -= learning_rate * output_error[i] * hidden.back()[j];
        }
        if (use_biases)
            biases.back()[i] -= learning_rate * output_error[i];
    }

    // Update weights and biases for hidden layers
    for (size_t i = hid_lens.size(); i-- > 0;)
    {
        for (size_t j = 0; j < hid_lens[i]; ++j)
        {
            for (size_t k = 0; k < ((i == 0) ? inp_len : hid_lens[i - 1]); ++k)
            {
                weights[i][k * hid_lens[i] + j] -= learning_rate * hidden_errors[i][j] * ((i == 0) ? neural_input[k] : hidden[i - 1][k]);
            }
            if (use_biases)
                biases[i][j] -= learning_rate * hidden_errors[i][j];
        }
    }
}

uint8_t NeuralNetwork::save(const std::string &file)
{
    std::ofstream out(file, std::ios::binary);
    if (!out)
    {
        std::cerr << "Unable to open file for saving." << std::endl;
        return 1;
    }

    // Save the network architecture
    uint16_t layers = hid_lens.size() + 2; // input, hidden layers, output
    out.write(reinterpret_cast<char *>(&layers), sizeof(layers));

    uint16_t input_size = static_cast<uint16_t>(inp_len);
    out.write(reinterpret_cast<char *>(&input_size), sizeof(input_size));

    for (const auto &size : hid_lens)
    {
        uint16_t layer_size = static_cast<uint16_t>(size);
        out.write(reinterpret_cast<char *>(&layer_size), sizeof(layer_size));
    }

    uint16_t output_size = static_cast<uint16_t>(out_len);
    out.write(reinterpret_cast<char *>(&output_size), sizeof(output_size));

    // Save the use_biases flag
    out.write(reinterpret_cast<const char *>(&use_biases), sizeof(use_biases));

    // Save weights and biases
    for (const auto &layer_weights : weights)
    {
        for (const auto &weight : layer_weights)
        {
            out.write(reinterpret_cast<const char *>(&weight), sizeof(weight));
        }
    }

    if (use_biases)
    {
        for (const auto &layer_biases : biases)
        {
            for (const auto &bias : layer_biases)
            {
                out.write(reinterpret_cast<const char *>(&bias), sizeof(bias));
            }
        }
    }

    out.close();
    return 0;
}

uint8_t NeuralNetwork::load(const std::string &file)
{
    // Load weights and biases from file
    std::ifstream in(file, std::ios::binary);
    if (!in)
    {
        std::cerr << "Unable to open file for loading." << std::endl;
        return 1;
    }

    // Load the network architecture
    uint16_t layers;
    in.read(reinterpret_cast<char *>(&layers), sizeof(layers));

    uint16_t input_size;
    in.read(reinterpret_cast<char *>(&input_size), sizeof(input_size));
    inp_len = static_cast<size_t>(input_size);

    hid_lens.resize(layers - 2);

    // Read hidden layer sizes
    for (size_t i = 0; i < hid_lens.size(); ++i)
    {
        uint16_t layer_size;
        in.read(reinterpret_cast<char *>(&layer_size), sizeof(layer_size));
        hid_lens[i] = static_cast<size_t>(layer_size);
    }

    uint16_t output_size;
    in.read(reinterpret_cast<char *>(&output_size), sizeof(output_size));
    out_len = static_cast<size_t>(output_size);

    // Load the use_biases flag
    in.read(reinterpret_cast<char *>(&use_biases), sizeof(use_biases));

    // Reinitialize weights and biases based on loaded architecture
    size_t prev_len = inp_len;
    weights.clear();
    biases.clear();

    for (size_t len : hid_lens)
    {
        weights.push_back(std::vector<float>(prev_len * len));
        if (use_biases)
            biases.push_back(std::vector<float>(len));
        prev_len = len;
    }
    weights.push_back(std::vector<float>(prev_len * out_len));
    if (use_biases)
        biases.push_back(std::vector<float>(out_len));

    hidden.resize(hid_lens.size());
    for (size_t i = 0; i < hid_lens.size(); ++i)
    {
        hidden[i].resize(hid_lens[i]);
    }
    output.resize(out_len);
    output_error.resize(out_len);

    // Load weights
    for (auto &layer_weights : weights)
    {
        for (auto &weight : layer_weights)
        {
            in.read(reinterpret_cast<char *>(&weight), sizeof(weight));
        }
    }

    // Load biases if use_biases is true
    if (use_biases)
    {
        for (auto &layer_biases : biases)
        {
            for (auto &bias : layer_biases)
            {
                in.read(reinterpret_cast<char *>(&bias), sizeof(bias));
            }
        }
    }

    in.close();

    return 0;
}
