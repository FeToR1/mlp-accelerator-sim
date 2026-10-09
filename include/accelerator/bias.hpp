#pragma once

#include "accelerator/dma.hpp"
#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(Bias) {
    SC_CTOR(Bias, simulator::Transactions & transactions,
            std::size_t lane_count)
        : transactions(transactions), values(lane_count) {}

    void load(DMA & dma, std::uint64_t address, std::size_t count);
    void apply(std::vector<float> & sums);
    std::size_t size_bytes() const { return values.size() * sizeof(float); }

   private:
    simulator::Transactions& transactions;
    std::vector<float> values;
    std::size_t output_count = 0;
};

}
