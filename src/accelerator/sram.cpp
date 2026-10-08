#include "accelerator/sram.hpp"

#include <algorithm>
#include <stdexcept>

namespace accelerator {

void SRAM::write(const std::vector<float>& values) {
    if (values.size() > memory.size()) {
        throw std::out_of_range("SRAM block exceeds capacity");
    }
    std::copy(values.begin(), values.end(), memory.begin());
}

std::vector<float> SRAM::read(std::size_t count) const {
    if (count > memory.size()) {
        throw std::out_of_range("SRAM read exceeds capacity");
    }
    return {memory.begin(), memory.begin() + count};
}

}
