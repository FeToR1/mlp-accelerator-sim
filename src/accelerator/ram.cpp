#include "accelerator/ram.hpp"

#include <algorithm>
#include <stdexcept>

namespace accelerator {

std::uint64_t ExternalRAM::allocate(std::size_t count) {
    constexpr std::size_t alignment_bytes = 16;
    constexpr std::size_t alignment_words = alignment_bytes / sizeof(float);
    if (count == 0) {
        throw std::invalid_argument("Cannot allocate an empty RAM block");
    }
    const auto padding =
        (alignment_words - memory.size() % alignment_words) % alignment_words;
    const auto available = memory.max_size() - memory.size();
    if (padding > available || count > available - padding) {
        throw std::length_error("RAM allocation is too large");
    }
    const auto start = memory.size() + padding;
    memory.resize(start + count, 0.0f);
    return static_cast<std::uint64_t>(start) * sizeof(float);
}

std::size_t ExternalRAM::block_start(std::uint64_t address,
                                     std::size_t count) const {
    if (address % sizeof(float) != 0) {
        throw std::invalid_argument("RAM address must be aligned to float");
    }
    const auto start = address / sizeof(float);
    if (start > memory.size() || count > memory.size() - start) {
        throw std::out_of_range("RAM access is outside allocated memory");
    }
    return static_cast<std::size_t>(start);
}

void ExternalRAM::write(std::uint64_t address,
                        const std::vector<float>& values) {
    const auto start = block_start(address, values.size());
    std::copy(values.begin(), values.end(), memory.begin() + start);
}

std::vector<float> ExternalRAM::read(std::uint64_t address,
                                     std::size_t count) const {
    const auto start = block_start(address, count);
    return {memory.begin() + start, memory.begin() + start + count};
}

}
