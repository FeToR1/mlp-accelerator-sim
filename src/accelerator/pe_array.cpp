#include "accelerator/pe_array.hpp"

#include <algorithm>
#include <stdexcept>

namespace accelerator {

PEArray::PEArray(sc_core::sc_module_name name, const AcceleratorConfig& config,
                 simulator::Transactions& transactions)
    : sc_core::sc_module(name), pes("pe"), config(config) {
    const auto local_k_capacity =
        (std::size_t(config.k_tile_size) + config.pe_count - 1) /
        config.pe_count;
    pes.init(config.pe_count, [&](const char* pe_name, std::size_t) {
        return new PE(pe_name, transactions, config.output_tile_size,
                      config.microbatch_size, local_k_capacity);
    });
}

void PEArray::clear_acc() {
    for (auto& pe : pes) {
        pe.clear_acc();
    }
}

void PEArray::load(DMA& dma, const std::vector<float>& input,
                   const std::vector<float>& weights, std::size_t batch_count) {
    if (batch_count == 0 || batch_count > std::size_t(config.microbatch_size) ||
        input.size() % batch_count != 0) {
        throw std::invalid_argument("PE array input must fit microbatch size");
    }
    const auto k_count = input.size() / batch_count;
    const auto lane_count = std::size_t(config.output_tile_size);
    if (k_count > std::size_t(config.k_tile_size) ||
        weights.size() != k_count * lane_count) {
        throw std::invalid_argument("PE array blocks must fit one K tile");
    }
    active_count = std::min(pes.size(), k_count);
    for (std::size_t pe_id = 0; pe_id < active_count; ++pe_id) {
        std::vector<float> local_input;
        std::vector<float> local_weights;
        for (std::size_t m = 0; m < batch_count; ++m) {
            for (std::size_t k = pe_id; k < k_count; k += pes.size()) {
                local_input.push_back(input[m * k_count + k]);
            }
        }
        for (std::size_t k = pe_id; k < k_count; k += pes.size()) {
            for (std::size_t lane = 0; lane < lane_count; ++lane) {
                local_weights.push_back(weights[k * lane_count + lane]);
            }
        }
        auto& pe = pes[pe_id];
        dma.write(pe.input_sram, local_input);
        dma.write(pe.weight_sram, local_weights);
        pe.load(pe.input_sram.read(local_input.size()),
                pe.weight_sram.read(local_weights.size()), batch_count);
    }
}

void PEArray::compute() {
    for (std::size_t pe_id = 0; pe_id < active_count; ++pe_id) {
        pes[pe_id].compute();
    }
}

}
