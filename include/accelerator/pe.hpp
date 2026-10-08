#pragma once

#include <sysc/kernel/sc_module.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(PE) {
    SC_CTOR(PE, simulator::Transactions& transactions, std::size_t lane_count)
        : acc(lane_count, 0.0f), transactions(transactions),
          lane_count(lane_count) {}

    void load(const std::vector<float>& input, const std::vector<float>& weights);
    void compute();

    std::vector<float> acc;
    std::uint64_t mac_count = 0;

   private:
    simulator::Transactions& transactions;
    const std::size_t lane_count;
    std::vector<float> x;
    std::vector<float> w;
};

}
