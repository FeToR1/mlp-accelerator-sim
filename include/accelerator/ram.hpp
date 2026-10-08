#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace accelerator {

class ExternalRAM {
    std::vector<float> memory;

    std::size_t block_start(std::uint64_t address, std::size_t count) const;

   public:
    std::uint64_t allocate(std::size_t count);
    void write(std::uint64_t address, const std::vector<float>& values);
    std::vector<float> read(std::uint64_t address, std::size_t count) const;

    std::size_t size_bytes() const { return memory.size() * sizeof(float); }
};

}
