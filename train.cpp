#include "neural.h"

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::string train_images_file = "train-images.idx3-ubyte";
std::string train_labels_file = "train-labels.idx1-ubyte";
std::string test_images_file = "t10k-images.idx3-ubyte";
std::string test_labels_file = "t10k-labels.idx1-ubyte";

void read_image_file_header(std::ifstream &file, int &magic_number, int &number_of_images, int &rows, int &cols)
{
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char *)&rows, sizeof(rows));
    rows = reverseInt(rows);
    file.read((char *)&cols, sizeof(cols));
    cols = reverseInt(cols);
}

void read_label_file_header(std::ifstream &file, int &magic_number, int &number_of_labels)
{
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverseInt(number_of_labels);
}

void load_image(std::ifstream &file, std::vector<uint8_t> &image)
{
    file.read((char *)image.data(), image.size());
}

void load_label(std::ifstream &file, uint8_t &label)
{
    file.read((char *)&label, sizeof(label));
}

void convert_image_to_input(const std::vector<uint8_t> &image, std::vector<float> &input)
{
    for (size_t i = 0; i < image.size(); ++i)
    {
        input[i] = image[i] / 255.0f;
    }
}

void convert_label_to_target(uint8_t label, std::vector<float> &target, size_t output_len)
{
    target.assign(output_len, 0.0f);
    target[label] = 1.0f;
}

#define TRAINING_SAMPLES 60000
#define TESTING_SAMPLES 10000
#define TRAINING_ROUNDS 100

int main()
{
    std::cout << "Start" << std::endl;

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const size_t input_len = 784;
    std::vector<size_t> hidden_lens = {300, 100, 50, 16}; // Example hidden layer sizes
    const size_t output_len = 10;
    const float learning_rate = 0.001f;

    NeuralNetwork nn(input_len, hidden_lens, output_len, true);

    std::ifstream train_images(train_images_file, std::ios::binary);
    std::ifstream train_labels(train_labels_file, std::ios::binary);
    std::ifstream test_images(test_images_file, std::ios::binary);
    std::ifstream test_labels(test_labels_file, std::ios::binary);

    if (!train_images.is_open() || !train_labels.is_open() || !test_images.is_open() || !test_labels.is_open())
    {
        std::cerr << "Unable to open one of the data files." << std::endl;
        return 1;
    }

    int magic_number, number_of_images, rows, cols, number_of_labels;
    read_image_file_header(train_images, magic_number, number_of_images, rows, cols);
    read_label_file_header(train_labels, magic_number, number_of_labels);

    std::vector<uint8_t> image(input_len);
    std::vector<float> neural_input(input_len);
    std::vector<float> target(output_len);

    uint8_t label = 10;

    if (nn.load("mnist.nn") == 1){
        std::cout << "Randomizing Weights and Biases" << std::endl;
        nn.initializeWeightsAndBiases();
    }

    for (size_t round = 0; round < TRAINING_ROUNDS; ++round)
    {

        // Accuracy Test Start
        test_images.seekg(16);
        test_labels.seekg(8);

        int correct = 0;
        for (size_t i = 0; i < TESTING_SAMPLES; ++i)
        {
            load_image(test_images, image);
            load_label(test_labels, label);

            convert_image_to_input(image, neural_input);
            nn.forward(neural_input);

            uint8_t prediction = std::distance(nn.output.begin(), std::max_element(nn.output.begin(), nn.output.end()));
            if (prediction == label)
            {
                ++correct;
            }
        }

        std::cout << "\nTesting completed with accuracy: " << (correct / float(TESTING_SAMPLES)) * 100.0f << "%" << std::endl;
        // Accuracy Test End

        std::cout << "Training round " << round + 1 << std::endl;

        train_images.seekg(16);
        train_labels.seekg(8);

        for (size_t i = 0; i < TRAINING_SAMPLES; ++i)
        {
            if (i % 1000 == 0)
                std::cout << "\rSample " << i + 1;
            load_image(train_images, image);
            load_label(train_labels, label);

            convert_image_to_input(image, neural_input);
            convert_label_to_target(label, target, output_len);

            nn.learn(neural_input, target, learning_rate);
        }

        nn.save("mnist.nn");
    }

    std::cout << "Training completed!" << std::endl;

    return 0;
}
