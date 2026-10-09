#include "accelerator/sigmoid_lut.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace accelerator {

SigmoidLUT::SigmoidLUT(sc_core::sc_module_name name,
                       simulator::Transactions& transactions,
                       std::size_t lane_count)
    : sc_core::sc_module(name),
      transactions(transactions),
      lane_count(lane_count) {
    for (std::size_t i = 0; i < table.size(); ++i) {
        const auto value = min_value + float(i) / points_per_unit;
        table[i] = 1.0f / (1.0f + std::exp(-value));
    }
}

float SigmoidLUT::lookup(float value) const {
    if (std::isnan(value)) {
        return value;
    }
    if (value <= min_value) {
        return 0.0f;
    }
    if (value >= max_value) {
        return 1.0f;
    }
    const auto position = (value - min_value) * points_per_unit;
    const auto index = std::min(std::size_t(position), table.size() - 2);
    const auto fraction = position - float(index);
    return table[index] + fraction * (table[index + 1] - table[index]);
}

void SigmoidLUT::apply(std::vector<float>& sums, std::size_t output_count) {
    if (output_count == 0 || output_count > lane_count || sums.empty() ||
        sums.size() % lane_count != 0) {
        throw std::invalid_argument("Sigmoid block must fit output tile");
    }
    transactions.transfer(simulator::TransactionType::BiasToLut,
                          sums.size() * sizeof(float));
    // LUT без продвижения model time
    for (std::size_t m = 0; m < sums.size() / lane_count; ++m) {
        for (std::size_t lane = 0; lane < output_count; ++lane) {
            auto& value = sums[m * lane_count + lane];
            value = lookup(value);
        }
    }
}

}
