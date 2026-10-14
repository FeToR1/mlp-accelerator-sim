#include "accelerator/model_file.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace accelerator {
namespace {

void read_values(std::ifstream& file, float* values, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!(file >> values[i]) || !std::isfinite(values[i])) {
            throw std::runtime_error(
                "Missing or invalid FP32 value in model file");
        }
    }
}

void check_end(std::ifstream& file) {
    file >> std::ws;
    if (!file.eof()) {
        throw std::runtime_error("Unexpected data at end of model file");
    }
}

}

std::vector<FCLayer> load_weights(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open weights: " + path);
    std::string format;
    int count = 0;
    if (!(file >> format >> count) || format != "MLP_WEIGHTS_V1" ||
        count <= 0) {
        throw std::runtime_error("Invalid weights header");
    }
    std::vector<FCLayer> layers;
    for (int i = 0; i < count; ++i) {
        int k = 0, n = 0;
        if (!(file >> k >> n) || k <= 0 || n <= 0 ||
            (!layers.empty() && k != layers.back().weights.cols())) {
            throw std::runtime_error("Invalid or mismatched layer dimensions");
        }
        FCLayer layer{WeightMatrix(k, n), std::vector<float>(n)};
        read_values(file, layer.weights.data(), layer.weights.size());
        read_values(file, layer.bias.data(), layer.bias.size());
        layers.push_back(std::move(layer));
    }
    check_end(file);
    return layers;
}

WeightMatrix load_input(const std::string& path, int width) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open input: " + path);
    std::string format;
    int m = 0, k = 0;
    if (!(file >> format >> m >> k) || format != "MLP_INPUT_V1" || m <= 0 ||
        k != width) {
        throw std::runtime_error(
            "Input must have shape [batch, first layer width]");
    }
    WeightMatrix input(m, k);
    read_values(file, input.data(), input.size());
    check_end(file);
    return input;
}

}
