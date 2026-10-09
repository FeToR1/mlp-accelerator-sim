#pragma once

#include <sysc/kernel/sc_module.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "accelerator/sram.hpp"
#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(PE) {
    SC_CTOR(PE, simulator::Transactions & transactions, std::size_t lane_count,
            std::size_t microbatch_size, std::size_t local_k_capacity)
        : input_sram("input_sram", microbatch_size * local_k_capacity),
          weight_sram("weight_sram", local_k_capacity * lane_count),
          acc(microbatch_size * lane_count, 0.0f),
          transactions(transactions),
          lane_count(lane_count),
          microbatch_size(microbatch_size) {}

    void load(const std::vector<float>& input,
              const std::vector<float>& weights, std::size_t batch_count,
              std::size_t output_count);
    void clear_acc();
    void compute();

    SRAM input_sram;
    SRAM weight_sram;
    std::vector<float> acc;
    std::uint64_t mac_count = 0;

   private:
    simulator::Transactions& transactions;
    const std::size_t lane_count;
    const std::size_t microbatch_size;
    std::size_t batch_count = 0;
    std::size_t k_count = 0;
    std::size_t output_count = 0;
    std::vector<float> x;
    std::vector<float> w;
};

}
