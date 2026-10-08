#include "accelerator/dma.hpp"

#include <stdexcept>

namespace accelerator {

std::vector<float> DMA::read(std::uint64_t address, std::size_t count) {
    transactions.transfer(simulator::TransactionType::RamToDma,
                          count * sizeof(float));
    return ram.read(address, count);
}

void DMA::load(std::uint64_t address, std::size_t count, SRAM& destination) {
    const auto block = read(address, count);
    transactions.transfer(simulator::TransactionType::DmaToSram,
                          block.size() * sizeof(float));
    destination.write(block);
}

void DMA::store(std::uint64_t address, std::size_t count, const SRAM& source) {
    const auto& values = source.read();
    if (count > values.size()) {
        throw std::out_of_range("SRAM read exceeds capacity");
    }
    transactions.transfer(simulator::TransactionType::SramToDma,
                          count * sizeof(float));
    const std::vector<float> block(values.begin(), values.begin() + count);
    transactions.transfer(simulator::TransactionType::DmaToRam,
                          block.size() * sizeof(float));
    ram.write(address, block);
}

}
