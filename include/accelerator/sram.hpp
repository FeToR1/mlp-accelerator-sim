#pragma once

#include <sysc/kernel/sc_module.h>

#include <cstddef>
#include <vector>

namespace accelerator {

SC_MODULE(SRAM) {
    SC_CTOR(SRAM, std::size_t count) : memory(count) {}

    void write(const std::vector<float>& values);
    const std::vector<float>& read() const { return memory; }
    std::size_t size_bytes() const { return memory.size() * sizeof(float); }

   private:
    std::vector<float> memory;
};

}
