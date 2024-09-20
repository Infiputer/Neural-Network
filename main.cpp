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

void clear_screen()
{
    // Clear screen using ANSI escape codes
    std::cout << "\033[2J\033[1;1H";
}

int main()
{
    std::cout << "Start" << std::endl;

    NeuralNetwork nn;

    std::ifstream test_images(test_images_file, std::ios::binary);
    std::ifstream test_labels(test_labels_file, std::ios::binary);

    if (!test_images.is_open() || !test_labels.is_open())
    {
        std::cerr << "Unable to open one of the data files." << std::endl;
        return 1;
    }

    int magic_number, number_of_images, rows, cols, number_of_labels;

    std::vector<uint8_t> image(784);
    std::vector<float> neural_input(784);
    std::vector<float> target(10);

    uint8_t label = 10;

    if (nn.load("mnist.nn") == 1)
    {
        std::cout << "'mnist.nn' file cannot by opened. Trying running train.cpp first." << std::endl;
	exit(1);
    }

    // Accuracy Test Start
    test_images.seekg(16);
    test_labels.seekg(8);

    int correct = 0;
    for (size_t i = 0; i < 500; ++i)
    {
        load_image(test_images, image);
        load_label(test_labels, label);

        convert_image_to_input(image, neural_input);
        nn.forward(neural_input);

        uint8_t prediction = std::distance(nn.output.begin(), std::max_element(nn.output.begin(), nn.output.end()));
        if (true || prediction != label)
        {
            clear_screen();

            for (uint8_t y_axis = 0; y_axis < 28; y_axis++)
            {
                for (uint8_t x_axis = 0; x_axis < 28; x_axis++)
                {
                    std::cout << ((image[y_axis * 28 + x_axis] > 128) ? 'X' : ((image[y_axis * 28 + x_axis] > 50) ? '+' : ' '));
                }
                std::cout << '\n';
            }

            for (uint8_t number = 0; number < 10; number++)
            {
                std::cout << "Number " << (int)number << ": " << std::round(nn.output[number] * 10000.0f) / 100 << '%' << std::endl;
            }
            std::cout << "\nPress Enter to continue...";
            std::cin.get();
        }
    }

    nn.save("mnist.nn");

    return 0;
}
