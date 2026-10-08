#pragma once

#include <cstdint>

namespace accelerator {

struct FCCommand {
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;

    std::uint64_t x_addr = 0;
    std::uint64_t w_addr = 0;
    std::uint64_t bias_addr = 0;
    std::uint64_t y_addr = 0;
};

}
