#include "accelerator/pe.hpp"

#include <algorithm>
#include <stdexcept>

namespace accelerator {

void PE::load(const std::vector<float>& input,
              const std::vector<float>& weights, std::size_t batch_count,
              std::size_t output_count) {
    if (batch_count == 0 || batch_count > microbatch_size) {
        throw std::invalid_argument("PE batch must fit microbatch size");
    }
    if (output_count == 0 || output_count > lane_count) {
        throw std::invalid_argument("PE outputs must fit MAC lanes");
    }
    if (input.size() % batch_count != 0 ||
        weights.size() != input.size() / batch_count * lane_count) {
        throw std::invalid_argument("PE blocks must match M x K and K x lanes");
    }
    transactions.transfer(simulator::TransactionType::InputSramToPe,
                          input.size() * sizeof(float));
    x = input;
    transactions.transfer(simulator::TransactionType::WeightSramToPe,
                          weights.size() * sizeof(float));
    w = weights;
    this->batch_count = batch_count;
    k_count = input.size() / batch_count;
    this->output_count = output_count;
}

void PE::clear_acc() { std::fill(acc.begin(), acc.end(), 0.0f); }

void PE::compute() {
    // MAC без продвижения model time
    for (std::size_t k = 0; k < k_count; ++k) {
        for (std::size_t m = 0; m < batch_count; ++m) {
            for (std::size_t lane = 0; lane < output_count; ++lane) {
                acc[m * lane_count + lane] +=
                    x[m * k_count + k] * w[k * lane_count + lane];
                ++mac_count;
            }
        }
    }
}

}
