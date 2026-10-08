#include "accelerator/reduction.hpp"

#include <stdexcept>
#include <utility>

namespace accelerator {

std::vector<float> Reduction::reduce(const sc_core::sc_vector<PE>& pes,
                                     std::size_t count) {
    if (pes.size() == 0) {
        throw std::invalid_argument("Reduction needs at least one PE");
    }
    std::vector<std::vector<float>> partials;
    for (const auto& pe : pes) {
        if (count > pe.acc.size()) {
            throw std::out_of_range("Reduction block exceeds ACC capacity");
        }
        transactions.transfer(simulator::TransactionType::PeToReduction,
                              count * sizeof(float));
        partials.emplace_back(pe.acc.begin(), pe.acc.begin() + count);
    }

    // Сложения без продвижения model time
    while (partials.size() > 1) {
        std::vector<std::vector<float>> next;
        for (std::size_t i = 0; i < partials.size(); i += 2) {
            auto sum = std::move(partials[i]);
            if (i + 1 < partials.size()) {
                for (std::size_t j = 0; j < count; ++j) {
                    sum[j] += partials[i + 1][j];
                }
            }
            next.push_back(std::move(sum));
        }
        partials = std::move(next);
    }
    return std::move(partials.front());
}

}
