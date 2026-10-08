#include "accelerator/pe.hpp"

#include <stdexcept>

namespace accelerator {

void PE::load(const std::vector<float>& input,
              const std::vector<float>& weights) {
    if (weights.size() != input.size() * lane_count) {
        throw std::invalid_argument("PE weights must contain K x lanes values");
    }
    transactions.transfer(simulator::TransactionType::InputSramToPe,
                          input.size() * sizeof(float));
    x = input;
    transactions.transfer(simulator::TransactionType::WeightSramToPe,
                          weights.size() * sizeof(float));
    w = weights;
}

void PE::compute() {
    // MAC без продвижения model time
    for (std::size_t k = 0; k < x.size(); ++k) {
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            acc[lane] += x[k] * w[k * lane_count + lane];
            ++mac_count;
        }
    }
}

}
