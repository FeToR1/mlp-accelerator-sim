#pragma once

#include "accelerator/ram.hpp"
#include "accelerator/sram.hpp"
#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(DMA) {
    SC_CTOR(DMA, ExternalRAM & ram, simulator::Transactions & transactions)
        : ram(ram), transactions(transactions) {}

    std::vector<float> read(std::uint64_t address, std::size_t count);
    void write(SRAM & destination, const std::vector<float>& values);
    void load(std::uint64_t address, std::size_t count, SRAM& destination);
    void store(std::uint64_t address, std::size_t count, const SRAM& source,
               std::size_t source_offset = 0);

   private:
    ExternalRAM & ram;
    simulator::Transactions& transactions;
};

}
