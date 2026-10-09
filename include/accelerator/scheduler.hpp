#pragma once

#include <vector>

#include "accelerator/command.hpp"
#include "accelerator/config.hpp"
#include "accelerator/dma.hpp"
#include "accelerator/pe_array.hpp"
#include "accelerator/reduction.hpp"

namespace accelerator {

SC_MODULE(Scheduler) {
    SC_CTOR(Scheduler, const AcceleratorConfig& config, DMA& dma,
            PEArray& array, Reduction& reduction)
        : config(config), dma(dma), array(array), reduction(reduction) {}

    std::vector<float> execute(const FCCommand& command);

   private:
    const AcceleratorConfig config;
    DMA & dma;
    PEArray & array;
    Reduction & reduction;
};

}
