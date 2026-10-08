#pragma once

#include <sysc/kernel/sc_module.h>

#include <cstdint>
#include <vector>

#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(PE) {
    SC_CTOR(PE, simulator::Transactions& transactions)
        : transactions(transactions) {}

    void load(const std::vector<float>& input, const std::vector<float>& weights);
    void compute();

    float acc = 0.0f;
    std::uint64_t mac_count = 0;

   private:
    simulator::Transactions& transactions;
    std::vector<float> x;
    std::vector<float> w;
};

}
