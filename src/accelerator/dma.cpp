#include "accelerator/dma.hpp"

#include <sysc/kernel/sc_wait.h>

namespace accelerator {

std::vector<float> DMA::read(std::uint64_t address, std::size_t count) {
    sc_core::wait(transfer_time);
    return ram.read(address, count);
}

}
