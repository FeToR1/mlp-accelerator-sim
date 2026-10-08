#pragma once

#include <sysc/kernel/sc_time.h>

#include "accelerator/ram.hpp"
#include "accelerator/sram.hpp"

namespace accelerator {

SC_MODULE(DMA) {
    SC_CTOR(DMA, ExternalRAM& ram) : ram(ram) {}

    std::vector<float> read(std::uint64_t address, std::size_t count);
    void load(std::uint64_t address, std::size_t count, SRAM& destination);

    const sc_core::sc_time transfer_time{1, sc_core::SC_NS};

   private:
    ExternalRAM& ram;
};

}
