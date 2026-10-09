#include "accelerator/program.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace accelerator {

MLPProgram load_mlp(ExternalRAM& ram, const WeightMatrix& input,
                    const std::vector<FCLayer>& layers,
                    const AcceleratorConfig& config) {
    if (input.rows() <= 0 || input.cols() <= 0 || layers.empty()) {
        throw std::invalid_argument("MLP needs input and at least one layer");
    }
    auto width = input.cols();
    auto max_width = width;
    for (const auto& layer : layers) {
        if (layer.weights.rows() != width || layer.weights.cols() <= 0 ||
            layer.bias.size() != std::size_t(layer.weights.cols())) {
            throw std::invalid_argument("MLP layer dimensions do not match");
        }
        width = layer.weights.cols();
        max_width = std::max(max_width, width);
    }
    const auto buffer_count = std::size_t(input.rows()) * max_width;
    auto x_addr = ram.allocate(buffer_count);
    auto y_addr = ram.allocate(buffer_count);
    ram.write(x_addr, {input.data(), input.data() + input.size()});

    MLPProgram program;
    for (const auto& layer : layers) {
        const auto packed = pack_weights(layer.weights, config);
        FCCommand command;
        command.m = input.rows();
        command.k = layer.weights.rows();
        command.n = layer.weights.cols();
        command.x_addr = x_addr;
        command.y_addr = y_addr;
        command.w_addr = ram.allocate(packed.values.size());
        command.bias_addr = ram.allocate(layer.bias.size());
        ram.write(command.w_addr, packed.values);
        ram.write(command.bias_addr, layer.bias);
        program.commands.push_back(command);
        std::swap(x_addr, y_addr);
    }
    return program;
}

}
