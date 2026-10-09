#pragma once

#include "accelerator/command_queue.hpp"
#include "accelerator/program.hpp"

namespace accelerator {

SC_MODULE(Accelerator) {
    SC_CTOR(Accelerator, const AcceleratorConfig& config)
        : config(config),
          ram("ram"),
          dma("dma", ram, transactions),
          array("array", config, transactions),
          reduction("reduction", transactions),
          bias("bias", transactions, config.output_tile_size),
          lut("lut", transactions, config.output_tile_size),
          output_sram("output_sram", std::size_t(config.microbatch_size) *
                                         config.output_tile_size),
          scheduler("scheduler", config, dma, array, reduction, bias, lut,
                    output_sram),
          queue("queue", scheduler, transactions) {}

    MLPProgram load(const WeightMatrix& input,
                    const std::vector<FCLayer>& layers) {
        return load_mlp(ram, input, layers, config);
    }

    const AcceleratorConfig config;
    simulator::Transactions transactions;
    ExternalRAM ram;
    DMA dma;
    PEArray array;
    Reduction reduction;
    Bias bias;
    SigmoidLUT lut;
    SRAM output_sram;
    Scheduler scheduler;
    CommandQueue queue;
};

}
