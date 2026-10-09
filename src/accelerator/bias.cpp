#include "accelerator/bias.hpp"

#include <algorithm>
#include <stdexcept>

namespace accelerator {

void Bias::load(DMA& dma, std::uint64_t address, std::size_t count) {
    if (count == 0 || count > values.size()) {
        throw std::invalid_argument("Bias block must fit output tile");
    }
    const auto block = dma.read(address, count);
    transactions.transfer(simulator::TransactionType::DmaToBias,
                          count * sizeof(float));
    std::copy(block.begin(), block.end(), values.begin());
    output_count = count;
}

void Bias::apply(std::vector<float>& sums) {
    if (output_count == 0 || sums.empty() || sums.size() % values.size() != 0) {
        throw std::invalid_argument("Load bias before applying to output tile");
    }
    transactions.transfer(simulator::TransactionType::ReductionToBias,
                          sums.size() * sizeof(float));
    // Bias без продвижения model time
    for (std::size_t m = 0; m < sums.size() / values.size(); ++m) {
        for (std::size_t lane = 0; lane < output_count; ++lane) {
            sums[m * values.size() + lane] += values[lane];
        }
    }
}

}
