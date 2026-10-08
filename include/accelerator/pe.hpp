#pragma once

#include <sysc/kernel/sc_module.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(PE) {
    SC_CTOR(PE, simulator::Transactions& transactions, std::size_t lane_count,
            std::size_t microbatch_size)
        : acc(microbatch_size * lane_count, 0.0f), transactions(transactions),
          lane_count(lane_count), microbatch_size(microbatch_size) {}

    void load(const std::vector<float>& input, const std::vector<float>& weights,
              std::size_t batch_count);
    void clear_acc();
    void compute();

    std::vector<float> acc;
    std::uint64_t mac_count = 0;

   private:
    simulator::Transactions& transactions;
    const std::size_t lane_count;
    const std::size_t microbatch_size;
    std::size_t batch_count = 0;
    std::size_t k_count = 0;
    std::vector<float> x;
    std::vector<float> w;
};

}
