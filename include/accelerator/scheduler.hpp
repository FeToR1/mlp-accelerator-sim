#pragma once

#include "accelerator/bias.hpp"
#include "accelerator/command.hpp"
#include "accelerator/config.hpp"
#include "accelerator/dma.hpp"
#include "accelerator/pe_array.hpp"
#include "accelerator/reduction.hpp"
#include "accelerator/sigmoid_lut.hpp"
#include "accelerator/sram.hpp"

namespace accelerator {

SC_MODULE(Scheduler) {
    SC_CTOR(Scheduler, const AcceleratorConfig& config, DMA& dma,
            PEArray& array, Reduction& reduction, Bias& bias, SigmoidLUT& lut,
            SRAM& output_sram)
        : config(config),
          dma(dma),
          array(array),
          reduction(reduction),
          bias(bias),
          lut(lut),
          output_sram(output_sram) {}

    void execute(const FCCommand& command);

   private:
    const AcceleratorConfig config;
    DMA & dma;
    PEArray & array;
    Reduction & reduction;
    Bias & bias;
    SigmoidLUT & lut;
    SRAM & output_sram;
};

}
