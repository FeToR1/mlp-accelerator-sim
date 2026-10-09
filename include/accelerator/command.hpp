#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>

namespace accelerator {

struct FCCommand {
    static constexpr std::size_t descriptor_bytes = 64;
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;

    std::uint64_t x_addr = 0;
    std::uint64_t w_addr = 0;
    std::uint64_t bias_addr = 0;
    std::uint64_t y_addr = 0;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const FCCommand& command) {
    return stream << "FC M=" << command.m << " N=" << command.n
                  << " K=" << command.k;
}

}
