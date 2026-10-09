#pragma once

#include <vector>

#include "accelerator/bias.hpp"
#include "accelerator/command.hpp"
#include "accelerator/config.hpp"
#include "accelerator/dma.hpp"
#include "accelerator/pe_array.hpp"
#include "accelerator/reduction.hpp"
#include "accelerator/sigmoid_lut.hpp"

namespace accelerator {

SC_MODULE(Scheduler) {
    SC_CTOR(Scheduler, const AcceleratorConfig& config, DMA& dma,
            PEArray& array, Reduction& reduction, Bias& bias, SigmoidLUT& lut)
        : config(config),
          dma(dma),
          array(array),
          reduction(reduction),
          bias(bias),
          lut(lut) {}

    std::vector<float> execute(const FCCommand& command);

   private:
    const AcceleratorConfig config;
    DMA & dma;
    PEArray & array;
    Reduction & reduction;
    Bias & bias;
    SigmoidLUT & lut;
};

}
