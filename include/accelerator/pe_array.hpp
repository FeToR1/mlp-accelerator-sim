#pragma once

#include <sysc/utils/sc_vector.h>

#include "accelerator/config.hpp"
#include "accelerator/dma.hpp"
#include "accelerator/pe.hpp"

namespace accelerator {

SC_MODULE(PEArray) {
    SC_CTOR(PEArray, const AcceleratorConfig& config,
            simulator::Transactions& transactions);

    void clear_acc();
    void load(DMA& dma, const std::vector<float>& input,
              const std::vector<float>& weights, std::size_t batch_count);
    void compute();

    sc_core::sc_vector<PE> pes;

   private:
    const AcceleratorConfig config;
    std::size_t active_count = 0;
};

}
