#include "accelerator/dma.hpp"

#include <stdexcept>

namespace accelerator {

std::vector<float> DMA::read(std::uint64_t address, std::size_t count) {
    transactions.transfer(simulator::TransactionType::RamToDma,
                          count * sizeof(float));
    return ram.read(address, count);
}

void DMA::write(SRAM& destination, const std::vector<float>& values) {
    transactions.transfer(simulator::TransactionType::DmaToSram,
                          values.size() * sizeof(float));
    destination.write(values);
}

void DMA::load(std::uint64_t address, std::size_t count, SRAM& destination) {
    const auto block = read(address, count);
    write(destination, block);
}

void DMA::store(std::uint64_t address, std::size_t count, const SRAM& source,
                std::size_t source_offset) {
    const auto& values = source.read();
    if (source_offset > values.size() ||
        count > values.size() - source_offset) {
        throw std::out_of_range("SRAM read exceeds capacity");
    }
    transactions.transfer(simulator::TransactionType::SramToDma,
                          count * sizeof(float));
    const std::vector<float> block(values.begin() + source_offset,
                                   values.begin() + source_offset + count);
    transactions.transfer(simulator::TransactionType::DmaToRam,
                          block.size() * sizeof(float));
    ram.write(address, block);
}

}
