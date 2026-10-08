#include "accelerator/dma.hpp"

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

}
