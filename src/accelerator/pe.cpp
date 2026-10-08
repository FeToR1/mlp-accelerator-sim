#include "accelerator/pe.hpp"

#include <stdexcept>

namespace accelerator {

void PE::load(const std::vector<float>& input,
              const std::vector<float>& weights) {
    if (input.size() != weights.size()) {
        throw std::invalid_argument("PE input and weight sizes differ");
    }
    transactions.transfer(simulator::TransactionType::InputSramToPe,
                          input.size() * sizeof(float));
    x = input;
    transactions.transfer(simulator::TransactionType::WeightSramToPe,
                          weights.size() * sizeof(float));
    w = weights;
}

void PE::compute() {
    for (std::size_t k = 0; k < x.size(); ++k) {
        acc += x[k] * w[k];
        ++mac_count;
    }
}

}
