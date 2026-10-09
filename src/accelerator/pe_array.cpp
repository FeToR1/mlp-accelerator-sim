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
                   std::uint64_t weight_tile_addr, std::size_t batch_count,
                   std::size_t output_count) {
    if (batch_count == 0 || batch_count > std::size_t(config.microbatch_size) ||
        input.size() % batch_count != 0) {
        throw std::invalid_argument("PE array input must fit microbatch size");
    }
    const auto k_count = input.size() / batch_count;
    const auto lane_count = std::size_t(config.output_tile_size);
    if (output_count == 0 || output_count > lane_count) {
        throw std::invalid_argument(
            "PE array outputs must fit one output tile");
    }
    if (k_count > std::size_t(config.k_tile_size)) {
        throw std::invalid_argument("PE array blocks must fit one K tile");
    }
    const auto local_k_capacity =
        (std::size_t(config.k_tile_size) + pes.size() - 1) / pes.size();
    const auto weight_block_count = local_k_capacity * lane_count;
    for (auto& pe : pes) {
        pe.mac_capacity +=
            std::uint64_t(config.microbatch_size) * weight_block_count;
    }
    active_count = std::min(pes.size(), k_count);
    for (std::size_t pe_id = 0; pe_id < active_count; ++pe_id) {
        std::vector<float> local_input;
        for (std::size_t m = 0; m < batch_count; ++m) {
            for (std::size_t k = pe_id; k < k_count; k += pes.size()) {
                local_input.push_back(input[m * k_count + k]);
            }
        }
        auto& pe = pes[pe_id];
        dma.write(pe.input_sram, local_input);
        const auto weight_addr =
            weight_tile_addr + pe_id * weight_block_count * sizeof(float);
        dma.load(weight_addr, weight_block_count, pe.weight_sram);
        const auto valid_weight_count =
            local_input.size() / batch_count * lane_count;
        pe.load(pe.input_sram.read(local_input.size()),
                pe.weight_sram.read(valid_weight_count), batch_count,
                output_count);
    }
}

void PEArray::compute() {
    for (std::size_t pe_id = 0; pe_id < active_count; ++pe_id) {
        pes[pe_id].compute();
    }
}

}
