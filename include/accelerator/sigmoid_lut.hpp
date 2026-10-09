#pragma once

#include <sysc/kernel/sc_module.h>

#include <array>
#include <cstddef>
#include <vector>

#include "accelerator/sram.hpp"
#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(SigmoidLUT) {
    static constexpr float min_value = -16.0f;
    static constexpr float max_value = 16.0f;
    static constexpr std::size_t interval_count = 2048;
    static constexpr std::size_t table_size = interval_count + 1;
    static constexpr float points_per_unit =
        interval_count / (max_value - min_value);

    SC_CTOR(SigmoidLUT, simulator::Transactions & transactions,
            std::size_t lane_count);

    float lookup(float value) const;
    void apply(const std::vector<float>& sums, std::size_t output_count,
               SRAM& output);
    std::size_t size_bytes() const { return table.size() * sizeof(float); }

   private:
    simulator::Transactions& transactions;
    const std::size_t lane_count;
    std::array<float, table_size> table;
};

}
