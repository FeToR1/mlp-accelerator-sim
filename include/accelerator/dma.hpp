#pragma once

#include "accelerator/ram.hpp"
#include "accelerator/sram.hpp"
#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(DMA) {
    SC_CTOR(DMA, ExternalRAM& ram, simulator::Transactions& transactions)
        : ram(ram), transactions(transactions) {}

    std::vector<float> read(std::uint64_t address, std::size_t count);
    void load(std::uint64_t address, std::size_t count, SRAM& destination);

   private:
    ExternalRAM& ram;
    simulator::Transactions& transactions;
};

}
